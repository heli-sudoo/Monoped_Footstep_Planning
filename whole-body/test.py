#!/usr/bin/env python3
from pinocchio.visualize import GepettoVisualizer as Visualizer
from models import *

robot = load_monoped_model(verbose=True)
q0 = robot.q0
model = robot.model
data = robot.data

q0 = robot.q0

robot.setVisualizer(Visualizer())
robot.initViewer()
robot.loadViewerModel("pinocchio")
robot.display(q0)

print("robot.nv = ", robot.nv)
print("robot.nq = ", robot.nq)