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
        axl.set_ylabel('GRF (N)')
        axl.tick_params(axis='y', labelcolor='orange')        

        axr = axl.twinx() # instantiate a second axes that shares the same x-axis
        axr.plot(time, traj.cxs, color='green',label='px')
        axr.plot(time, traj.cys, color='green',label='py',linestyle='dashed')
        axr.set_ylabel('Foot Position (m)')
        axr.tick_params(axis='y', labelcolor='green')

        fname = 'contactforceandposition.png'
        fig.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(fname)
        
        
# Solve footstep planning with SMT solver          
smtSol, Traj = smt_monoped.Footstepplan_smt()
Horizon = len(Traj)
print('Number of steps of smt generated trjaectory %s' % (Horizon))
# Get initial guess from smt solver
uinit = [np.array([Traj.fxs[i], Traj.fys[i], Traj.cxs[i], Traj.cys[i]]) for i in range(Horizon)]

# Create differential monoped model
monopedDAM = Monoped.DifferentialActionModelMonoped()

# Create monoped action data 
monopedData = monopedDAM.createData()

dt = 0.1
# Create discrete-time monoped model using Euler integration
monopedIAM = crocoddyl.IntegratedActionModelEuler(monopedDAM, dt)

# Initial condition
x0 = np.array([0,1,0,0,0,0])

# Create terminal action model
tmonopedDAM = Monoped.DifferentialActionModelMonoped()
tmonopedIAM = crocoddyl.IntegratedActionModelEuler(monopedDAM)

tmonopedDAM.costWeights = np.array([100,100,100,
                                    50,50,50,
                                    50,50,
                                    0,0])

# Create shooting problem
problem = crocoddyl.ShootingProblem(x0, [monopedIAM] * Horizon, tmonopedIAM)

state_init = problem.rollout(uinit)

for i in range(Horizon):
    Traj.xs[i] = state_init[i][0]
    Traj.ys[i] = state_init[i][1]
    Traj.Rs[i] = state_init[i][2]

plot_trajectory(Traj, 'control')

smtSol.plot_solution(Traj, save=True)





