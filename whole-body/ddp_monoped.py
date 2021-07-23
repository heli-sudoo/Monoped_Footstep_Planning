#!/usr/bin/python3

from monoped_utils import *
from models import *

robot = load_monoped_model()
HoppingGait = MonopedHoppingProblem(robot.model)
# problem = HoppingGait.createHoppingProblem()
HoppingGait.checkDimension()