#!/usr/bin/env python

##
#
# Use the z3 SMT solver for footstep planning for a simple planar
# monoped, including the nonlinear centroidal dynamics. 
#
##

import z3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle, Polygon

def to_float(num):
    """
    Convert a rational number from z3 (e.g. RatNumRef) to a 
    standard python float.
    """
    return float(num.numerator_as_long())/float(num.denominator_as_long())

class MonopedState():
    """
    A simple object representing the state of the monoped at a given time
    instant. 
    """
    def __init__(self, x, y, xd, yd, fx, fy, cx, cy, R, w):
        """
        Construct the state. 

        @param x   horizontal position of the CoM
        @param y   vertical position of the CoM
        @param xd  horizontal velocity of the CoM
        @param yd  vertical velocity of the CoM
        @param fx  contact force horizontal component
        @param fy  contact force vertical component
        @param cx  contact point horizontal position
        @param cy  contact point vertical position
        @param R   rotation of the body about the CoM
        @param w   angular velocity of the body
        """
        self.x = x
        self.y = y
        self.xd = xd
        self.yd = yd
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.R = R
        self.w = w

    def __str__(self):
        return "MonopedState with (x, y) = %0.2f, %0.2f" % (self.x, self.y)

class MonopedTrajectory():
    """
    Represent a trajectory of the monoped. This is merely syntatic sugar 
    for a list of MonopedStates. 
    """
    def __init__(self, xs, ys, xds, yds, fxs, fys, cxs, cys, Rs, ws):
        """
        Store the given data as a list of MonopedStates.

        @param xs   horizontal positions of the CoM
        @param ys   vertical positions of the CoM
        @param xds  horizontal velocities of the CoM
        @param yds  vertical velocities of the CoM
        @param fxs  contact force horizontal components
        @param fys  contact force vertical components
        @param cxs  contact point horizontal positions
        @param cys  contact point vertical positions
        @param Rs   rotations of the body about the CoM
        @param ws   angular velocities of the body
        """
        self.N = len(xs)
        assert len(ys) == self.N, "Length of ys does not match length of xs"
        assert len(xds) == self.N, "Length of xds does not match length of xs"
        assert len(yds) == self.N, "Length of yds does not match length of xs"
        assert len(fxs) == self.N, "Length of fxs does not match length of xs"
        assert len(fys) == self.N, "Length of fys does not match length of xs"
        assert len(cxs) == self.N, "Length of cxs does not match length of xs"
        assert len(cys) == self.N, "Length of cys does not match length of xs"
        assert len(Rs) == self.N, "Length of Rs does not match length of xs"
        assert len(ws) == self.N, "Length of ws does not match length of xs"
        
        # Direct access (e.g. x0 = Traj.xs[0])
        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)
        self.xds = np.asarray(xds)
        self.yds = np.asarray(yds)
        self.fxs = np.asarray(fxs)
        self.fys = np.asarray(fys)
        self.cxs = np.asarray(cxs)
        self.cys = np.asarray(cys)
        self.Rs = np.asarray(Rs)
        self.ws = np.asarray(ws)

        # List access (e.g. x0 = Traj[0].x)
        self.lst = []
        for i in range(self.N):
            self.lst.append(
                    MonopedState(xs[i], ys[i], 
                                 xds[i], yds[i], 
                                 fxs[i], fys[i],
                                 cxs[i], cys[i],
                                 Rs[i], ws[i]))

    def __str__(self):
        return "%s step MonopedTrajectory: (%0.1f, %0.1f) --> (%0.1f, %0.1f)" \
                % (self.N, self.lst[0].x, self.lst[0].y, self.lst[-1].x, self.lst[-1].y)

    def __getitem__(self, i):
        return self.lst[i]

    def __add__(self, other):
        """
        Override the '+' operator so we can add two MonopedTrajectory objects.
        """
        xs = np.hstack([self.xs, other.xs])
        ys = np.hstack([self.ys, other.ys])
        xds = np.hstack([self.xds, other.xds])
        yds = np.hstack([self.yds, other.yds])
        fxs = np.hstack([self.fxs, other.fxs])
        fys = np.hstack([self.fys, other.fys])
        cxs = np.hstack([self.cxs, other.cxs])
        cys = np.hstack([self.cys, other.cys])
        Rs = np.hstack([self.Rs, other.Rs])
        ws = np.hstack([self.ws, other.ws])

        return MonopedTrajectory(xs, ys, xds, yds, fxs, fys, cxs, cys, Rs, ws)

    def __len__(self):
        return self.N

    def pop(self):
        """
        Remove the MonopedState in this trajectory and return it.
        """
        self.xs = self.xs[:-1]
        self.ys = self.ys[:-1]
        self.xds = self.xds[:-1]
        self.yds = self.yds[:-1]
        self.fxs = self.fxs[:-1]
        self.fys = self.fys[:-1]
        self.cxs = self.cxs[:-1]
        self.cys = self.cys[:-1]
        self.Rs = self.Rs[:-1]
        self.ws = self.ws[:-1]
        self.N -= 1

        return self.lst.pop()

class MonopedSMTSolver():
    """
    A class for creating and solving footstep planning problems for a simple
    monoped using the z3 SMT solver. 
    """
    def __init__(self, m=1, g=9.81, dt=0.1, mu=0.5, r=0.3, I=np.array([[1]]), 
            y_min=0.5, y_max=1.5, xd_min=-1.5, xd_max=1.5, yd_min=-1.5, yd_max=1.5, 
            cd_min=-1.5, cd_max=1.5, gap=[1.5,2.0]):
        """
        Set parameters for this optimization problem and initialize the solver. 

        @param m        mass of the robot's body
        @param g        gravitational acceleration
        @param dt       timestep used in the (forward Euler) integration scheme
        @param mu       friction coefficient
        @param r        size of the contact point bounding box
        @param I        Intertia matrix
        @param y_min    minimum height of the CoM
        @param y_max    maximum height of the CoM
        @param xd_min   minimum horizontal velocity of the CoM
        @param xd_max   maximum horizontal velocity of the CoM
        @param yd_min   minimum vertical velocity of the CoM
        @param yd_max   maximum vertical velocity of the CoM
        @param cd_min   minimum velocity of the foot
        @param cd_max   maximum velocity of the foot
        @param gap      start and end positions of the gap

        """
        self.m = m
        self.g = g
        self.dt = dt
        self.mu = mu
        self.r = r
        self.I = I
        self.y_min = y_min
        self.y_max = y_max
        self.xd_min = xd_min
        self.xd_max = xd_max
        self.yd_min = yd_min
        self.yd_max = yd_max
        self.cd_min = cd_min
        self.cd_max = cd_max
        self.gap = gap

        # Initialize solver
        self.Refresh()

    def Refresh(self):
        """
        Re-initialize the solver, which resets all of the constraints
        that may have previously been placed. 
        """
        # Note: this is in theory possible in a more elegant way with
        # the s.push() and s.pop() operators, but for reasons I don't 
        # quite understand z3's incremental solver doesn't perform as
        # well as the vanilla version on certain problems. 
        # (see https://stackoverflow.com/questions/26416814/)
        z3.set_param("parallel.enable",True)
        self.s = z3.Solver()
        
        # Create lists to store results
        self.xs = []; self.xds = []       # CoM position
        self.ys = []; self.yds = []
        self.fxs = []; self.fys = []      # contact force
        self.cxs = []; self.cys = []      # contact point position
        self.Rs = []; self.ws = []        # rotation and angular velocity

    def SetInitialConditions(self, S):
        """
        Set constraints for the given initial conditions. 
        This allows us to use the results of a prior solve
        as a starting point for a subsequent solve. 

        @param S  a MonopedState representing the initial condition.
        """
        # Create variables
        x0, y0 = z3.Reals('x0 y0')       # position of the CoM
        xd0, yd0 = (S.xd,S.yd)           # velocity of the CoM
        fx0, fy0 = z3.Reals('fx0 fy0')   # contact forces
        cx0, cy0 = z3.Reals('cx0 cy0')   # contact force position
        R0 = z3.Real('R0')               # rotation about CoM
        w0 = S.w                          # angular velocity

        # Add constraints
        self.s.add( x0 == S.x )   # start position
        self.s.add( y0 == S.y )

        self.s.add( cx0 == S.cx ) # contact point
        self.s.add( cy0 == S.cy )

        self.s.add( R0 == S.R )  # start at zero rotation
            
        # Contact and friction constraints should apply at these initial
        # conditions as well
        self.AddContactConstraints(cx0, cy0, 0, 0, fx0, fy0)
        self.AddGapConstraint(cx0, fx0, fy0)
        self.AddFrictionConeConstraint(fx0, fy0)

        # update stored variable lists
        self.xs.append(x0); self.xds.append(xd0)
        self.ys.append(y0); self.yds.append(yd0)
        self.fxs.append(fx0); self.fys.append(fy0)
        self.cxs.append(cx0); self.cys.append(cy0)
        self.Rs.append(R0); self.ws.append(w0)

    def SetRunConstraints(self, N, x_des, y_des):
        """
        Add constraints for the dynamics of the monoped over a trajectory
        of N steps, ending at (x_des, y_des)

        @param N      the number of steps in the trajectory
        @param x_des  the desired final x-position of the CoM
        @param y_des  the desired final y-position of the CoM
        """
        for k in range(1,N+1):
            # Declare new variables for this timestep
            xk, yk = z3.Reals('x%s y%s' % (k,k))       # CoM position
            fxk, fyk = z3.Reals('fx%s fy%s' % (k,k))   # Contact force
            cxk, cyk = z3.Reals('cx%s cy%s' % (k,k) )  # Contact position
            Rk = z3.Real('R%s' % k)  # rotation about CoM

            # Compute CoM and contact point velocity (forward Euler)
            xdk = (xk - self.xs[-1]) / self.dt
            ydk = (yk - self.ys[-1]) / self.dt

            cxdk = (cxk - self.cxs[-1]) / self.dt
            cydk = (cyk - self.cys[-1]) / self.dt
            
            # Apply linear dynamics constraints
            xdd = (xdk - self.xds[-1]) / self.dt
            ydd = (ydk - self.yds[-1]) / self.dt
            self.s.add([ self.m*xdd == self.fxs[-1] ])                  # f = ma in x direction
            self.s.add([ self.m*ydd == self.fys[-1] - self.m*self.g ])  # f = ma - mg in y direction

            # Apply angular dynamics constraints
            r_com = np.array([cxk-xk, cyk-yk])   # position of contact point relative to CoM
            f = np.array([fxk, fyk])             # contact force vector
            hdot = np.cross(r_com, f)            # angular momentum dot

            wdot = np.linalg.inv(self.I)*hdot    # angular velocity dot. (hdot = I*wdot + w x (Iw))
            wk = (Rk - self.Rs[-1]) / self.dt         # angular velocity
            wdk = (wk - self.ws[-1]) / self.dt
            self.s.add([ wdot[0,0] == wdk ])

            # Limits on rotation about CoM
            self.AddRotationConstraints(Rk)

            # Constrain position and velocity of the CoM
            self.AddCoMPositionConstraints(xk, yk)
            self.AddCoMVelocityConstraints(xdk, ydk)

            # Constrain position and velocity of the foot (relative to the CoM)
            self.AddFootPositionConstraints(xk, yk, cxk, cyk)
            self.AddFootVelocityConstraints(cxdk, cydk)

            # Contact and friction constraints
            self.AddContactConstraints(cxk, cyk, cxdk, cydk, fxk, fyk)
            self.AddGapConstraint(cxk, fxk, fyk)
            self.AddFrictionConeConstraint(fxk, fyk)

            # Add new variables to the list
            self.xs.append(xk); self.xds.append(xdk)
            self.ys.append(yk); self.yds.append(ydk)
            self.fxs.append(fxk); self.fys.append(fyk)
            self.cxs.append(cxk); self.cys.append(cyk)
            self.Rs.append(Rk); self.ws.append(wk)

        # Final conditions constraints
        self.s.add( self.xs[-1] == x_des )
        self.s.add( self.ys[-1] == y_des )
        self.s.add( self.Rs[-1] == 0 )

    def AddRotationConstraints(self, Rk):
        # Limits on rotation: note that these are nonlinear and can lead to
        # significant slowdown
        self.s.add( -np.pi/2 <= Rk )
        self.s.add( Rk <= np.pi/2 )

    def AddCoMPositionConstraints(self, xk, yk):
        # Limits on CoM height
        self.s.add( yk <= self.y_max )
        self.s.add( self.y_min <= yk )
           
    def AddFootPositionConstraints(self, xk, yk, cxk, cyk):
        # Constraint on contact position relative to CoM position
        self.s.add([ cxk <= xk + self.r ])
        self.s.add([ xk - self.r <= cxk ])

        self.s.add([ cyk <= self.r  ])
        self.s.add([ -self.r <= cyk ])

    def AddCoMVelocityConstraints(self, xdk, ydk):
        # Velocity limits on CoM
        self.s.add( xdk <= self.xd_max )
        self.s.add( self.xd_min <= xdk )
        self.s.add( ydk <= self.yd_max )
        self.s.add( self.yd_min <= ydk )

    def AddFootVelocityConstraints(self, cxdk, cydk):
        # Velocity limits on contact points
        self.s.add( cxdk <= self.cd_max )
        self.s.add( self.cd_min <= cxdk )
        self.s.add( cydk <= self.cd_max )
        self.s.add( self.cd_min <= cydk )

    def AddContactConstraints(self, cxk, cyk, cxdk, cydk, fxk, fyk):
        # Force can only be applied when in contact, and contact
        # point is stationary
        self.s.add(z3.Xor(
                z3.And([cyk == 0, cxdk == 0, cydk == 0]),
                z3.And([fyk == 0, fxk == 0])
              ))
        
        # Contact point must always be above ground
        self.s.add( cyk >= 0 )

    def AddGapConstraint(self, cxk, fxk, fyk):
        # Force can't be applied when we're over the gap
        self.s.add(z3.Implies(
                z3.And([ self.gap[0] <= cxk, cxk <= self.gap[1] ]),
                z3.And([fyk == 0, fxk == 0])))

    def AddFrictionConeConstraint(self, fxk, fyk): 
        # Forces must lie in the friction cone
        self.s.add( fxk <= self.mu*fyk )
        self.s.add( -fxk <= self.mu*fyk )

    def Solve(self):
        """
        Solve the stored SMT problem and return the results
        """
        res = self.s.check()
        solver_time = self.s.statistics().time
        
        print(res)
        print("Solver Time: %ss" % solver_time)

        N = len(self.xs)
        if res == z3.CheckSatResult(True):
            # A solution exists: return it
            soln = self.s.model()

            # Extract numerical representation of solution
            xs = [to_float(soln[self.xs[i]]) for i in range(N)]
            ys = [to_float(soln[self.ys[i]]) for i in range(N)]

            xds = [(xs[i]-xs[i-1]) / self.dt for i in range(1,N)]
            yds = [(ys[i]-ys[i-1]) / self.dt for i in range(1,N)]
            xds.insert(0, self.xds[0])
            yds.insert(0, self.xds[0])
            
            fxs = [to_float(soln[self.fxs[i]]) for i in range(N-1)]  # no constraints on force at last 
            fys = [to_float(soln[self.fys[i]]) for i in range(N-1)]  # timestep, so solution is None
            fxs.append(0)  # add a filler value for the final contact force
            fys.append(0)

            cxs = [to_float(soln[self.cxs[i]]) for i in range(N)]
            cys = [to_float(soln[self.cys[i]]) for i in range(N)]

            Rs = [to_float(soln[self.Rs[i]]) for i in range(N)]
            ws = [self.ws[i] for i in range(N)]
      
            return MonopedTrajectory(xs,ys,xds,yds,fxs,fys,cxs,cys,Rs,ws), solver_time
        
        else:
            # No solution.
            return None, solver_time

    def plot_solution(self, soln, save=False):
        """
        Make a matplotlib animation of the given solution.

        @param soln  a MonopedTrajectory storing the solution
        @param save  (optional) whether to save the animation as an mp4
        """
        # set up axes
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # CoM marker
        com_point, = ax.plot([0],[0], 'ro', markersize=5)
        com_point.set_data(0,0)

        # Rectangulary body
        body_width=0.4
        body_height=0.25
        body = Rectangle([0-body_width/2,0-body_height/2],body_width, body_height, color="red", alpha=0.5)
        ax.add_patch(body)

        # Contact location markers
        cj_point, = ax.plot([0],[0], 'bo', markersize=5)

        # Friction cones
        co = [0,0]   # cone origin
        ch = 0.4       # cone height
        cone_data = np.array([co,[-ch*self.mu,ch],[ch*self.mu,ch]])
        cone = Polygon(cone_data, color="green", alpha=0.3)
        ax.add_patch(cone)

        # Contact location constraint
        cj_constraint = plt.Rectangle([0-self.r,0-self.r],2*self.r,2*self.r, color="blue", alpha=0.5)
        ax.add_patch(cj_constraint)
        
        # Contact force arrows
        fj_arrow, = ax.plot([0,0],[0,0.5], 'g-', linewidth=2)

        # Ground
        #plt.axhline(y=0, color="grey", linestyle="-")
        plt.plot([-100,self.gap[0]],[0,0], color="black")
        plt.plot([self.gap[1],100],[0,0], color="black")

        def data_gen():
            gen_list = (np.hstack([soln.xs[i], 
                                   soln.ys[i],
                                   soln.fxs[i],
                                   soln.fys[i],
                                   soln.cxs[i],
                                   soln.cys[i],
                                   soln.Rs[i],
                                   i*self.dt]) for i in range(soln.N))
            return gen_list

        def init():
            ax.axis('equal')
            
            ax.set_ylim(-0.5,2)
            ax.set_xlim(-0.5, 3.5)
            ax.set_title("t = 0")


        def run(data):
            x, y, fx, fy, cx, cy, R, t = data

            # Update CoM location
            com_point.set_data(x,y)

            # Update body position and orientation
            T = transforms.Affine2D().rotate_around(x,y,R) + ax.transData
            body.set_transform(T)
            body.set_xy([x-body_width/2, y-body_height/2])

            # Update contact location constraint
            cj_constraint.set_xy([x-self.r,0-self.r])

            # Update contact force location
            cj_point.set_data([cx],[cy])
       
            # Update friction cone
            co = [cx,0]   # cone origin
            ch = 0.4       # cone height
            cone.set_xy(np.array([co,[co[0]-ch*self.mu,ch],[co[0]+ch*self.mu,ch]]))

            # Update contact force vector
            scale_factor = 0.01
            fj_arrow.set_data([cx,cx+scale_factor*fx],[0,scale_factor*fy])

            # show current time in the title
            ax.set_title("t=%0.2f" % t)

        ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, interval=4000*self.dt)

        if save:
            ani.save("smt_monoped.gif")

        plt.show()

def add_to_trajectory(Traj, solver, N, N0=10, dt=0.1):
    """
    Solve the SMT problem to add an additional N steps to 
    the given trajectory. 

    If the SMT solver cannot find a satisfying segment, we remove the previous
    N steps and resolve for a segment with 2N steps. This process repeats recursively
    until we reach the beginning of the trajectory. 

    @param Traj   a MonopedTrajectory that we're adding to
    @param solver the MonopedSMTSolver that stores the problem specification
    @param N      number of timesteps to be considered in this segment
    @param N0     default number of timesteps per segment. Useful when we're in recursion.
    @param dt     discretization timestep, using for computing target x-position
    """
    print("Adding an an additional %s steps to a %s step trajectory" % (N, len(Traj)-1))
    solver.Refresh()

    X0 = Traj[-1]                    # Set initial condition as last point in given
    solver.SetInitialConditions(X0)  # trajectory.

    x_des = X0.x + N*dt
    solver.SetRunConstraints(N=N, x_des=x_des, y_des=1)

    new_traj, solve_time = solver.Solve()

    if new_traj is not None:
        # Add to the stored trajectory
        Traj.pop()          # remove last state in old trajectory since this is now redundant
        return Traj + new_traj
    else:
        if len(Traj) < N:
            print("Segment unsatisfiable from begining")
            # Note that we still need to increase trajectory from the end to 
            # prove unsatisfiablity of the specification overall
            return None

        # Remove the last N0 steps
        [Traj.pop() for i in range(N0)]
        N_new = N + N0

        return add_to_trajectory(Traj, solver, N_new, N0=N0, dt=dt)



def Footstepplan_smt():
    N_segs = 3   # number of segments
    N = 10       # length of each segment
    dt = 0.1     # timestep

    # Set up the solver
    solver = MonopedSMTSolver(dt=dt, gap=[0.8,1.2])

    # Set the initial condition
    x, y, xd, yd, fx, fy, cx, cy, R, w = (0,1,0,0,0,0,0,0,0,0)
    X0 = MonopedState(x,y,xd,yd,fx,fy,cx,cy,R,w)

    # Perform an initial solve
    solver.SetInitialConditions(X0)
    solver.SetRunConstraints(N=N, x_des=dt*N, y_des=1)
    Traj, time = solver.Solve()
     
    for i in range(N_segs):
        Traj = add_to_trajectory(Traj, solver, N, dt=0.1)

        if Traj is None:
            # We tried starting from the very beginning again
            # and the trajectory is unsatisfiable
            # TODO: note that we still need to add to the end
            # of the trajectory to have completeness
            break
   
    return solver, Traj

