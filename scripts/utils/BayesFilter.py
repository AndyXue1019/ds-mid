#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class BayesFilter:
    def __init__(self):
        # Motion model
        self.T = np.array(
            [[0.8, 0.1, 0.1],
             [0.1, 0.8, 0.1],
             [0.1, 0.1, 0.8]]
        )
        # Sensor model
        self.Z = np.array(
            [[0.9546, 0.0, 0.0454],
             [0.0, 0.9864, 0.0136],
             [0.0029, 0.0021, 0.9950]]
        )

    


