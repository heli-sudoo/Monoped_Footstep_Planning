#!/usr/bin/env python3

from re import U
import numpy as np
from scipy.interpolate import interp1d
import smt_monoped
import matplotlib.pyplot as plt

class SMT_Trajectory_Refiner():
    def __init__(self, Traj, dt_coarse, dt_fine):
        self.Traj = Traj
        self.dt_coarse = dt_coarse
        self.dt_fine = dt_fine
        self.N_coarse = len(Traj)
        self.N_interp = int(dt_coarse//dt_fine) # including the end points
        self.N_fine = (self.N_coarse-1)*(self.N_interp-2) + self.N_coarse

    def create_coarse_sequence(self):
        # create state and control list from SMT trajectory
        # each element of the list below contains only one step
        Xinit, Xref, Uinit, Ctact = [], [], [], []
        for i in range(self.N_coarse):          
            Xinit.append([self.Traj.xs[i], 
                          self.Traj.ys[i], 
                          self.Traj.Rs[i], 
                          self.Traj.xds[i], 
                          self.Traj.yds[i], 0])

            Xref.append([self.Traj.xs[i], 1, 0, 
                         self.Traj.xds[i], 0, 0])

            Uinit.append([self.Traj.fxs[i], 
                          self.Traj.fys[i]])

            Ctact.append([self.Traj.cxs[i], 
                          self.Traj.cys[i]])
            return Xinit, Xref, Uinit, Ctact
    
    def create_fine_sequence(self):
        # create state and control list from SMT trajectory
        # each element of the list below contains a sequence
        # of trajectory which is also a list
        X, Xr, U, C = [],[],[],[]
        Xphase, Xrphase, Uphase, Cphase = [],[],[],[]
        for i in range(self.N_coarse):
            x = [self.Traj.xs[i], 
                 self.Traj.ys[i], 
                 0, 
                 self.Traj.xds[i], 
                 0, 0]
            xr = [self.Traj.xs[i], 1, 0, 
                  0.9, 0, 0]
            u = [self.Traj.fxs[i], 
                 self.Traj.fys[i]]
            c = [self.Traj.cxs[i], 
                 self.Traj.cys[i]]
            Xphase.append(x)
            Xrphase.append(xr)
            Uphase.append(u)
            Cphase.append(c)                
            if i > 0:
                # if contact phase changes or the last step
                # terminate and stage phase
                if self.Traj.cys[i] != self.Traj.cys[i-1] or i==self.N_coarse-1:  
                    # change the last contact in phase to the initial contact of the same phase                
                    Cphase[-1] = Cphase[0]
                    # append phase trajectory to the library
                    X.append(Xphase)
                    Xr.append(Xrphase)
                    U.append(Uphase)
                    C.append(Cphase)
                    # clear phase trajectory and reset its initial condition
                    Xphase = [x]
                    Xrphase = [xr]
                    Uphase = [u]
                    Cphase = [c]
        Xfine, Xrfine, Ufine, Cfine = [],[],[],[]
        for p in range(len(X)):
            for i in range(len(X[p])-1):
                Xinterp = interp1d([1,self.N_interp], np.vstack((X[p][i], X[p][i+1])), axis=0)
                Xrinterp = interp1d([1,self.N_interp], np.vstack((Xr[p][i], Xr[p][i+1])), axis=0)
                if i==0:
                    Xfine_phase = Xinterp(np.arange(1,self.N_interp+1))
                    Xrfine_phase = Xrinterp(np.arange(1,self.N_interp+1))
                    Ufine_phase = np.tile(U[p][i],(self.N_interp,1))
                    Cfine_phase = np.tile(C[p][i],(self.N_interp,1))
                else:
                    Xfine_phase = np.vstack((Xfine_phase,Xinterp(np.arange(1,self.N_interp+1))))
                    Xrfine_phase = np.vstack((Xrfine_phase,Xrinterp(np.arange(1,self.N_interp+1))))
                    Ufine_phase = np.vstack((Ufine_phase,np.tile(U[p][i],(self.N_interp,1))))
                    Cfine_phase = np.vstack((Cfine_phase,np.tile(C[p][i],(self.N_interp,1))))                  
            Xfine.append(Xfine_phase)
            Xrfine.append(Xfine_phase)
            Ufine.append(Ufine_phase)
            Cfine.append(Cfine_phase)
        return Xfine, Xrfine, Ufine, Cfine

def plot_state(Xmuti, dt, save = False):
    X = Xmuti[0]
    for i in range(1, len(Xmuti)):
        X = np.vstack((X, Xmuti[i]))
    N = len(X)
    time = np.arange(N)*dt

    fig, axes = plt.subplots(1,2)
    axes[0].plot(time, X[:,0], color='orange')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('x (m)', color='orange')
    axes[0].tick_params(axis='y', labelcolor='orange')
    
    axes[1].plot(time, X[:,2], color='blue')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('theta (rad)')

    fname = 'state.png'
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig(fname)

def plot_contact(Umuti, Cmuti, dt, save = False):
    U = Umuti[0]
    C = Cmuti[0]
    for i in range(1, len(Umuti)):
        U = np.vstack((U, Umuti[i]))
        C = np.vstack((C, Cmuti[i]))
    N = len(U)
    time = np.arange(N)*dt
    fig, axl = plt.subplots()
    axl.plot(time, U[:,0], color='orange', label='fx')
    axl.plot(time, U[:,1], color='orange', label='fy',linestyle='dashed')
    axl.set_xlabel('Time (s)')
    axl.set_ylabel('GRF (N)', color='orange')
    axl.tick_params(axis='y', labelcolor='orange')        

    axr = axl.twinx() # instantiate a second axes that shares the same x-axis
    axr.plot(time, C[:,0], color='green',label='px')
    axr.plot(time, C[:,1], color='green',label='py',linestyle='dashed')
    axr.set_ylabel('Foot Position (m)',color='green')
    axr.tick_params(axis='y', labelcolor='green')

    fname = 'contactinfo.png'
    fig.legend()
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig(fname)

def plot_SMT_trajectory(traj, var, save=False):
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
        if save:
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
        if save:
            fig.savefig(fname)


