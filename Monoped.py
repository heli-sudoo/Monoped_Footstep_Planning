
import numpy as np
from numpy import arctan2, cos, sin
from numpy.core.numeric import Inf

import crocoddyl


class DifferentialActionModelMonoped(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, p, m=1, I=1):
        """ Action model for the Monoped (without legs).
        The transition model of an unicycle system is described as    
        params @ mode   : 'f' -> flight mode 's' -> stance mode  
        params @ p      : contact position 
        """
        """
        Joint q = np.array([xs, ys, the]) 
              xs    CoM position in x
              ys    CoM position in y
              the   Body orientaion
        Joint dev qd = np.array([xd,yd,thed])
        State x = np.hstack((q, qd))
        Control u = np.array([fx, fy]) is ground reaction force
        qdd = np.array([w-[0;g], cross(p-[x;y], u)/I])
        """        
        if p[1]>0: # flight phase
            nu = 0
            nr = 6      
            self.mode = 'f'  
        elif p[1]==0: # stance phase
            nu = 2
            nr = 8
            self.mode = 's'
        else:
            print('error: contact pos in y direction cannot be less than zero')
            return

        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(6), nu, nr) #nu = 2 or 0, $nr = 8 or 6

        self.m = m
        self.g = 9.81
        self.I = I
        self.wgrf = 5 # weight parameters of normal GRF penalty
        self.costWeights = [1,2,20,.01,.02,.01]
        if self.mode == 's':
            self.costWeights += [.1, .1]

        
        self.unone = np.zeros(self.nu)      
        self.nx = 6
        self.p = np.asarray(p)
        self.xd = np.zeros(self.nx)
        self.ud = np.zeros(nu)
    def set_ref(self, xd):
        self.xd = xd  
        if self.mode=='s':
            self.ud = np.array([0, self.g])        

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone  
        assert(self.nx == len(x))
        # Get control and foothold location              
        # Define dynamics equation (data.xout constains simply second-order EOM rather than state-space dyanmics)
        qdd = np.zeros(3)  
        qdd[:2] = np.array([0, -self.g])  
        if self.mode == 's':
            qdd[:2] += u 
            qdd[-1] = np.cross(self.p-x[0:2], u) /self.I
        
        data.xout = qdd
        # Define running cost
        z = np.concatenate((x-self.xd, u-self.ud))
        data.r = np.array(self.costWeights * (z**2))
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)))
        # Penalize normal ground reaction force in stance phase
        if self.mode=='s':
            data.cost += np.exp(-self.wgrf*u[-1])

    def calcDiff(model, data, x, u=None):     
        # Cost derivatives        
        nx = len(x)
        data.Lx = np.asarray((x-model.xd) * model.costWeights[:nx])
        np.fill_diagonal(data.Lxx, model.costWeights[:nx])
        if model.mode == 's':
            data.Lu = np.asarray((u-model.ud) * model.costWeights[nx:])         
            np.fill_diagonal(data.Luu, model.costWeights[nx:])
        
        # Dynamic derivatives 
        if model.mode=='f':
            u = np.zeros(2)      
        # data.Fx = np.vstack((np.hstack((np.zeros((3,3)),np.identity(3))),
        #                      np.zeros((2,6)),
        #                      np.array([-u[1],u[0],0,0,0,0])))
        data.Fx = np.vstack((np.zeros((2,6)),
                             np.array([-u[1],u[0],0,0,0,0])))
        if model.mode=='s':                                         
            data.Fu = np.vstack((np.identity(2),
                                 np.array([(model.p[1]-x[1]), -model.p[0]-x[0]])))
            data.Lu+=np.array([0, -model.wgrf*np.exp(-model.wgrf*u[-1])])
            data.Luu += np.array([[0,0], 
                                  [0, model.wgrf**2*np.exp(-model.wgrf*u[-1])]])
