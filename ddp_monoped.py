#!/usr/bin/env python3

import numpy as np
import crocoddyl
import Monoped
import smt_monoped
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import util_monoped

def create_differential_models(C, Xref):
    # This function creates running action model and terminal model for one phase
    assert len(C) == len(Xref), "length of contact not equal to length of reference"
    N = len(C)
    runningDAMs = [Monoped.DifferentialActionModelMonoped(C[i,:]) for i in range(N-1)]
    for i in range(N-1):
        runningDAMs[i].set_ref(Xref[i,:])
    terminalDAM = Monoped.DifferentialActionModelMonoped(C[-1,:])
    terminalDAM.set_ref(Xref[-1,:])    
    return runningDAMs, terminalDAM

def create_model_seqs(rDAMseqs, tDAMseqs, dt):
    rIAMseqs = []
    tIAMseqs = []
    n_seqs = len(rDAMseqs)
    for s in range(n_seqs):
        rIAMseqs.append([crocoddyl.IntegratedActionModelEuler(DAM, dt) for DAM in rDAMseqs[s]])
        tIAMseqs.append(crocoddyl.IntegratedActionModelEuler(tDAMseqs[s], 0))
    runningModels = []
    terminalModel= []
    for s in range(n_seqs-1):
        runningModels+=rIAMseqs[s]+[tIAMseqs[s]]
    runningModels+=rIAMseqs[-1]
    terminalModel = tIAMseqs[-1]
    return runningModels, terminalModel

def stack_vector_over_phase(Xmulti):
    Xbig = Xmulti[0]
    for i in range(1, len(Xmulti)):
        Xbig = np.vstack((Xbig, Xmulti[i]))
    return Xbig

# Solve footstep planning with SMT solver          
smtSol, Traj = smt_monoped.Footstepplan_smt()
Horizon = len(Traj)
print('Number of steps of smt generated trjaectory %s' % (Horizon))

# interpolate SMT trajectory
x0 = np.array([0,1,0,0,0,0]) # Initial condition
dt_coarse = 0.1
dt_fine = 0.01
refine = util_monoped.SMT_Trajectory_Refiner(Traj, dt_coarse, dt_fine)
xinit_fine, xref_fine, uinit_fine, ctact_fine = refine.create_fine_sequence()
n_phase = len(xinit_fine)
print("number of phase %s" % (n_phase))

# create shooting problem
runningDAMSeqs = []
terminalDAMSeqs = []
for i in range(n_phase):
    rDAMseq, tDAMseq, = create_differential_models(ctact_fine[i], xref_fine[i])    
    runningDAMSeqs.append(rDAMseq)
    terminalDAMSeqs.append(tDAMseq)
      
# modify cost
for i in range(n_phase):
    for rmodel in runningDAMSeqs[i]:
        rmodel.costWeights = [0,20,10,30,.1,.01]
        if rmodel.mode=='s':
            rmodel.costWeights+=[.1, .1]            
    terminalDAMSeqs[i].costWeights = [0,80,30,40,1,.1]
    if terminalDAMSeqs[i].mode=='s':
        terminalDAMSeqs[i].costWeights+=[0,0]
terminalDAMSeqs[-1].costWeights= [0,100,500,100,1,.1]

# create model sequences for shooting problem
runningmodels, terminalmodel = create_model_seqs(runningDAMSeqs, terminalDAMSeqs, dt_fine)
problem = crocoddyl.ShootingProblem(x0, runningmodels, terminalmodel)
ddp = crocoddyl.SolverFDDP(problem)

# formulate initial guess
X_fine = stack_vector_over_phase(xinit_fine)
U_fine = stack_vector_over_phase(uinit_fine)
C_fine = stack_vector_over_phase(ctact_fine)
N_fine = len(X_fine)
xs = crocoddyl.StdVec_VectorX()
us = crocoddyl.StdVec_VectorX()
for x in X_fine:
    xs.append(x)
for u in U_fine[:-1,:]:
    us.append(u)

# solve DDP
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
ddp.solve(xs,us,20)


# Plot trajectory and generate animation
xs_ddp, ys_ddp, xds_ddp, yds_ddp = [],[],[],[]
fxs_ddp, fys_ddp,cxs_ddp, cys_ddp =[],[],[],[]
Rs_ddp, ws_ddp=[],[]
for i in range(N_fine-1):
    xs_ddp.append(ddp.xs[i][0])
    ys_ddp.append(ddp.xs[i][1])
    xds_ddp.append(ddp.xs[i][3])
    yds_ddp.append(ddp.xs[i][4])
    fxs_ddp.append(ddp.us[i][0])
    fys_ddp.append(ddp.us[i][1])
    cxs_ddp.append(C_fine[i][0])
    cys_ddp.append(C_fine[i][1])
    Rs_ddp.append(ddp.xs[i][2])
    ws_ddp.append(ddp.xs[i][5])

Traj_ddp = smt_monoped.MonopedTrajectory(xs_ddp,ys_ddp,xds_ddp,yds_ddp,fxs_ddp,fys_ddp,cxs_ddp,cys_ddp,Rs_ddp,ws_ddp)
# Plot ddp solution
util_monoped.plot_SMT_trajectory(Traj_ddp, 'control')
util_monoped.plot_SMT_trajectory(Traj_ddp, 'state')
smtSol.plot_solution(Traj_ddp, filename="ddp_monoped.gif", save=True)





