#!/usr/bin/env python3

import numpy as np
import crocoddyl
import Monoped
import smt_monoped
import matplotlib.pyplot as plt


def plot_trajectory(traj, var):
    time = np.arange(len(traj.xs)) * 0.1
    if var=='control':
        fig, axl = plt.subplots()
        axl.plot(time, traj.fxs, color='orange', label='fx')
        axl.plot(time, traj.fys, color='orange', label='fy',linestyle='dashed')
        axl.set_xlabel('Time (s)')
        axl.set_ylabel('GRF (N)', color='orange')
        axl.tick_params(axis='y', labelcolor='orange')        

        axr = axl.twinx() # instantiate a second axes that shares the same x-axis
        axr.plot(time, traj.cxs, color='green',label='px')
        axr.plot(time, traj.cys, color='green',label='py',linestyle='dashed')
        axr.set_ylabel('Foot Position (m)',color='green')
        axr.tick_params(axis='y', labelcolor='green')

        fname = 'contactinfo.png'
        fig.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(fname)
    if var=='state':
        fig, axes = plt.subplots(1,2)
        axes[0].plot(time, traj.xs, color='orange')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('x (m)', color='orange')
        axes[0].tick_params(axis='y', labelcolor='orange')
        axr = axes[0].twinx()
        axr.plot(time, traj.ys, color='green')
        axr.set_ylabel('y (m)', color='green')
        axr.tick_params(axis= 'y', labelcolor='green')
        axr.set_ylim()

        axes[1].plot(time, traj.Rs, color='blue')
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('theta (rad)')

        fname = 'state.png'
        fig.tight_layout()
        plt.show()
        fig.savefig(fname)

        
# Solve footstep planning with SMT solver          
smtSol, Traj = smt_monoped.Footstepplan_smt()
Horizon = len(Traj)
print('Number of steps of smt generated trjaectory %s' % (Horizon))
# Get initial guess from smt solver
xinit = [np.array([Traj.xs[i], Traj.ys[i], Traj.Rs[i], Traj.xds[i], Traj.yds[i], 0]) for i in range(Horizon)]
xref = [np.array([Traj.xs[i], 1, 0, Traj.xds[i], 0, 0]) for i in range(Horizon)]
uinit = [np.array([Traj.fxs[i], Traj.fys[i]]) for i in range(Horizon)]
ctact = [np.array([Traj.cxs[i], Traj.cys[i]]) for i in range(Horizon)]


# Create differential monoped model
monopedDAMseq = [Monoped.DifferentialActionModelMonoped(ctact[i]) for i in range(Horizon-1)]
for i in range(Horizon-1):
    monopedDAMseq[i].set_ref(xref[i])

# Create monoped action data 
monopedDataseq = [monopedDAM.createData() for monopedDAM in monopedDAMseq]

dt = 0.1
# Create discrete-time monoped model using simpletic Euler integration
monopedIAMseq = [crocoddyl.IntegratedActionModelEuler(monopedDAM, dt) for monopedDAM in monopedDAMseq]

# # Numerical difference for sanity check
# monopedNDseq = [crocoddyl.DifferentialActionModelNumDiff(monopedDAM) for monopedDAM in monopedDAMseq]
# monopedIAMseq = [crocoddyl.IntegratedActionModelEuler(monopedND, dt) for monopedND in monopedNDseq]

# Initial condition
x0 = np.array([0,1,0,0,0,0])

# Create terminal action model
tmonopedDAM = Monoped.DifferentialActionModelMonoped(ctact[-1])
tmonopedIAM = crocoddyl.IntegratedActionModelEuler(tmonopedDAM, 0.)
tmonopedDAM.set_ref(xref[-1])

tmonopedDAM.costWeights = [.1,10,1,1,.1,.1]
if tmonopedDAM.mode=='s':
    tmonopedDAM.costWeights += [1, 1]                   

# Create shooting problem
problem = crocoddyl.ShootingProblem(x0, monopedIAMseq, tmonopedIAM)
ddp = crocoddyl.SolverFDDP(problem)

xs = crocoddyl.StdVec_VectorX()
us = crocoddyl.StdVec_VectorX()
for x in xinit:
    xs.append(x)
for i in range(Horizon-1):
    us.append(uinit[i])
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
ddp.solve(xs,us,100)

# Plot smt solution
smtSol.plot_solution(Traj, save=True)
plot_trajectory(Traj, 'control')
plot_trajectory(Traj, 'state')

# Plot ddp solution
for i in range(Horizon-1):
    Traj.xs[i] = ddp.xs[i][0]
    Traj.ys[i] = ddp.xs[i][1]
    Traj.Rs[i] = ddp.xs[i][2]        
    Traj.fxs[i] = ddp.us[i][0]
    Traj.fys[i] = ddp.us[i][1]
Traj.xs[-1] = ddp.xs[-1][0]   
Traj.ys[-1] = ddp.xs[-1][1]
Traj.Rs[-1] = ddp.xs[-1][2] 

plot_trajectory(Traj, 'control')
plot_trajectory(Traj, 'state')
smtSol.plot_solution(Traj, filename="ddp_monoped.gif", save=True)





