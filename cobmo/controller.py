"""Controller class definition."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


class Controller(object):
    """Controller object to store the model predictive control problem."""

    def __init__(
            self,
            conn,
            building
    ):
        """Initialize Controller object.

        - Obtain building model information
        - Create Pyomo problem
        """
        self.problem = pyo.ConcreteModel()
        self.solver = SolverFactory()
