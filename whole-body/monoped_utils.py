from numpy.core.arrayprint import set_string_function
from numpy.core.defchararray import equal, not_equal
from numpy.testing._private.utils import assert_equal
import crocoddyl
import pinocchio
from models import *

import numpy as np
class MonopedHoppingProblem:
    """ Defines monoped hopping problem
    This definitions follows the SimpleQuadrupedGaitProblem example here
    https://github.com/loco-3d/crocoddyl/blob/f9789cbf7ec6d32c08728d6b3038532593c96530/bindings/python/crocoddyl/utils/quadruped.py
    and other examples here
    https://github.com/loco-3d/crocoddyl/blob/master/examples/notebooks/whole_body_manipulation.ipynb
    :params rmodel: Pinnochio model of robot
    :params timeStep: integration time step in seconds
    """
    def __init__(self, rmodel):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Get food id given foot name defined in urdf file
        self.fId = self.rmodel.getFrameId('foot') 
        # Define default state
        q0 = self.rmodel.referenceConfigurations["stance"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])

    def createHoppingProblem(self, x0, timeStep, NKnots):
        """
        :param x0: initial state
        :param timeStep: integration time step
        :param NKnots: list of dict where each element contains the following info (for example the NKnots[0])
                       NKnots[0]['name'] represents phase name (stance, flying)
                       NKnots[0]['nknot'] represents the number of knots in this phase
        :return multi-phase monoped hopping problem
        """
        Models = []
        for i in range(len(NKnots)):
            nknots = NKnots[i]['nknot']
            if NKnots[i]['name'] == "stance":                
                rmodel_phase = [self.createStanceModel(timeStep=timeStep)] * (nknots - 1)
                tmodel_phase = [self.createStanceModel(0)]
            if NKnots[i]['name'] == "flying":
                rmodel_phase = [self.createFlyingModel(timeStep=timeStep)] * (nknots - 1)
                tmodel_phase = [self.createFlyingModel(0)]
            model_phase = rmodel_phase + tmodel_phase
            Models+=model_phase
        runningModels = Models[:-1]
        terminalModel = Models[-1]
        problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
        return problem

    def createStanceModel(self, timeStep, stateRef):
        """
        :param timeStep: integration time step
        :param stateRef: state reference used in state regularization x - xref
        :return either a running or a terminal action model for a stance phase
        """
        # Get the configuration, tangent, state, and control dimension
        nq, nv, nx = self.state.nq, self.state.nv, self.state.nx
        nu = self.actuation.nu
        # Build stance-phase model
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        # Add contact to stance-phase model
        xref = crocoddyl.FramePlacement(self.fId, np.array([0.,0.,0,]))
        pointContact = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu) #ContactModel6D for flat foot
        contactModel.addContact(self.rmodel.frames[self.fId].name+"_contact", pointContact)
        # Build cost models
        CostModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        # Define running and terminal cost
        # Use different weightings for running and terminal cost
        if timeStep != 0: # running cost
            stateWeights = np.array([0.1]*nq + [1]*nv)
            ctrlWeights = np.array([0.1]*nu)
        else: # terminal cost
            stateWeights = np.array([10]*nq + [10]*nv)
            ctrlWeights = np.zeros(nu)
        stateActivationModel = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
        ctrlActivationModel = crocoddyl.ActivationModelWeightedQuad(ctrlWeights ** 2)
        stateReg = crocoddyl.CostModelState(self.state, stateActivationModel, stateRef)
        ctrlReg = crocoddyl.CostModelCtrl(self.state, ctrlActivationModel, nu)
        # Add to cost model
        CostModel.addCost("state reg", stateReg, 1.0)
        CostModel.addCost("control reg", ctrlReg, 1.0)

        # Creating the action models for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     CostModel, 0., True)
        # Using timeStep=0 for terminal model would not propagate state and would only compute the cost
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)

        return model

    def createFlyingModel(self, timeStep, stateRef):
        """
        :param timeStep: integration time step
        :param stateRef: state reference used in state regularization x - xref
        :return action model for a swing phase    
        """
        nq, nv, nx = self.state.nq, self.state.nv, self.state.nx
        nu = self.actuation.nu
        # Build flight-phase dynamics model (adding no contact does the job)
        flyingModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        CostModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        # Define running and terminal cost
        # Use different weightings for running and terminal cost
        if timeStep != 0: # running cost
            stateWeights = np.array([0.1]*nq + [1]*nv)
            ctrlWeights = np.array([0.1]*nu)
        else: # terminal cost
            stateWeights = np.array([10]*nq + [10]*nv)
            ctrlWeights = np.zeros(nu)
        stateActivationModel = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
        ctrlActivationModel = crocoddyl.ActivationModelWeightedQuad(ctrlWeights ** 2)
        stateReg = crocoddyl.CostModelState(self.state, stateActivationModel, stateRef)
        ctrlReg = crocoddyl.CostModelCtrl(self.state, ctrlActivationModel, nu)
        # Add to cost model
        CostModel.addCost("state reg", stateReg, 1.0)
        CostModel.addCost("control reg", ctrlReg, 1.0)

        # Creating the action models for the flying dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, flyingModel,
                                                                      CostModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createImpulseModel(self, stateRef):
        """
        This function creates an action model for impulse dynamics at the end of a flight phase
        Definition of this function follows the example here
        https://github.com/loco-3d/crocoddyl/blob/f9789cbf7ec6d32c08728d6b3038532593c96530/unittest/factory/action.cpp
        :param stateRef: terminal state reference for a flight phase
        :return action model 
        """
        # Get the configuration, tangent, state, and control dimension
        nq, nv, nx = self.state.nq, self.state.nv, self.state.nx
        nu = self.actuation.nu
        # Build impulse model
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        # Impulse from point-contact foot
        pointFootImpulse = crocoddyl.ImpulseModel3D(self.state, self.fId)
        # Add foot impulse to impulse model (add another foot if any)
        impulseModel.addImpulse(pointFootImpulse)
        # Define terminal cost
        terminalCost = crocoddyl.CostModelSum(self.state, 0)
        stateWeights = np.array([10]*nq + [10]*nv)
        stateActivationModel = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
        stateReg = crocoddyl.CostModelState(self.state, stateActivationModel, stateRef)
        terminalCost.addCost("state reg", stateReg, 1.0)
        r_coeff = 0. # use default values 0. Not sure yet the funcionality of r_coeff and dampling
        damping = 0.
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, terminalCost, r_coeff, damping, True)

        return model

    def checkDimension(self):
        print(f"Dimension of config space nq = {self.state.nq}")
        assert(self.state.nq == 6), "Dimension of config space not equal to 6"
        print(f"Dimension of tangent space nv = {self.state.nv}")
        assert(self.state.nv == 5), "Dimension of config space not equal to 5"
        print(f"Dimension of control space nu = {self.actuation.nu}")
        assert(self.actuation.nu == 2), "Dimension of config space not equal to 2"

        

