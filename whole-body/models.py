#!/usr/bin/env python3
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import numpy as np
import os
from termcolor import colored


def load_monoped_fl_model(verbose=False):
    start_path = os.path.abspath(os.getcwd())
    URDF_PATH = "/monoped_description/monoped_fl.urdf"
    REF_POSTURE = "stance"

    robot = RobotWrapper.BuildFromURDF(start_path+URDF_PATH)
    q0 = np.asarray([0, 0, 0, 0, 0])
    robot.q0 = q0
    robot.model.referenceConfigurations[REF_POSTURE] = q0
    if verbose:
        print_model_usage(robot)
    return robot


def load_monoped_model(verbose=False):
    start_path = os.path.abspath(os.getcwd())
    URDF_PATH = "/monoped_description/monoped.urdf"
    REF_POSTURE = "stance"

    robot = RobotWrapper.BuildFromURDF(
        start_path+URDF_PATH, None, pin.JointModelPlanar())
    q0 = np.asarray([0, 0, 1, 0, 0, 0])
    robot.q0 = q0
    robot.model.referenceConfigurations[REF_POSTURE] = q0

    if verbose:
        print_model_usage(robot)

    return robot


def load_mini_cheetah(verbose=False):
    start_path = os.path.abspath(os.getcwd())
    URDF_PATH = "/mini_cheetah_description/mini_cheetah_mesh.urdf"
    #URDF_PATH = "/mini_cheetah_description/mini_cheetah_simple_v2.urdf"
    REF_POSTURE = "standing"

    robot = RobotWrapper.BuildFromURDF(
        start_path + URDF_PATH, [start_path], pinocchio.JointModelFreeFlyer(), verbose=True)
    if verbose:
        print_model_usage(robot)

    q0 = np.asarray([0.0, 0.0, 0.3,       # base position
                     1.0, 0.0, 0.0, 0.0,  # base orientation
                     0.0, -0.8, 1.6,       # left front (adab, hip, knee)
                     0.0, -0.8, 1.6,       # right front
                     0.0, -0.8, 1.6,       # left hind
                     0.0, -0.8, 1.6])      # right hind
    robot.q0 = q0
    robot.model.referenceConfigurations[REF_POSTURE] = q0

    return robot


def print_model_usage(robot):
    print(colored("robot.model usage:", "red"))
    for name, function in robot.model.__class__.__dict__.items():
        print(' **** %s: %s' % (name, function.__doc__))

    print(colored("robot.data usage:", "red"))
    for name, function in robot.data.__class__.__dict__.items():
        print(' **** %s: %s' % (name, function.__doc__))
