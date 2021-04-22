
import numpy as np
from numpy import arctan2, cos, sin

import crocoddyl


class DifferentialActionModelMonoped(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, m=1, dt=0.1, I=1):
        """ Action model for the Monoped (without legs).
        The transition model of an unicycle system is described as           
        """
        """
        Joint q = [x, y, theta]
        State x = [[q], [qd]]
        Constrol u = [w; p] where w = [fx; fy] is ground reaction force
        p = [px; py] is foot loction
        xd = [qd, w-[0;g]], cross(p-[x;y], u)/I]
        """        
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), 4, 10) #nu = 4, $nr = 10

        self.m = m
        self.dt = dt
        self.g = 9.81
        self.I = I
        self.costWeights = list(np.ones(self.nr))
        self.unone = np.zeros(self.nu)      
        self.nx = 6       
        

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone  
        assert(self.nx == len(x))
        # Get control and foothold location              
        w = u[0:2] # GRF
        p = u[2:4] # Foothold location
        # Define dynamics equation (data.xout constains simply second-order EOM rather than state-space dyanmics)
        qdd = np.zeros(3)
        qdd[:2] = w + np.array([0, -self.g])
        qdd[2] = np.cross(p-x[0:2], w) /self.I
        data.xout = qdd
        # Define running cost
        z = np.concatenate((x, u))
        data.r = np.array(self.costWeights * (z**2))
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)))

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        # Cost derivatives        
        nx = len(x)
        data.Lx = np.asarray(x * self.costWeights[:nx])
        data.Lu = np.asarray(u * self.costWeights[nx:]) 
        np.fill_diagonal(data.Lxx, self.costWeights[:nx])
        np.fill_diagonal(data.Luu, self.costWeights[nx:])
        

        # Dynamic derivatives
        w = u[:2] # GRF
        p = u[2:] # Foothold location
        data.Fx = np.vstack((np.hstack(np.zeros(3,3),np.identity(3)),
                             np.zeros(2,6),
                             np.array([-w[1],w[0],0,0,0,0])))
        data.Fu = np.vstack((np.zeros(3,4),
                             np.identity(2),
                             -(p[1]-x[1]),
                             p[0]-x[0]))
