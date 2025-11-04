#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class BayesFilter:
    def __init__(self, initial_belief: np.ndarray = None, robot: str = 'turtlebot'):
        # Motion model
        # 假設狀態不太會改變
        self.T = np.array(
            [
                [0.8, 0.1, 0.1],  # P(x_t|x_{t-1}=0)
                [0.1, 0.8, 0.1],  # P(x_t|x_{t-1}=1)
                [0.1, 0.1, 0.8],  # P(x_t|x_{t-1}=2)
            ]  
        )
        # Sensor model
        if robot == 'turtlebot':
            self.Z = np.array(
                [
                    [0.9950, 0.0021, 0.0029],  # P(z_t|x_t=0)
                    [0.0454, 0.9546,    0.0],  # P(z_t|x_t=1)
                    [0.0136,    0.0, 0.9864],  # P(z_t|x_t=2)
                ]  
            )
        else:
            self.Z = np.array(
                [
                    [   1.0,    0.0,    0.0],  # P(z_t|x_t=0)
                    [0.8593, 0.1407,    0.0],  # P(z_t|x_t=1)
                    [0.3983,    0.0, 0.6017],  # P(z_t|x_t=2)
                ]
            )

        if initial_belief is not None:
            self.belief = initial_belief
        else:
            self.belief = np.array([1 / 3, 1 / 3, 1 / 3])  # 初始信念均等分配

    def update(self, z_t_proba: np.ndarray):
        """
        執行一次完整的預測和更新循環。
        :param z_t_proba: 來自分類器的觀測機率分佈, e.g., np.array([0.1, 0.8, 0.1])
        """
        # 1. 預測步驟
        # bel'(x_t) = T^T * bel(x_{t-1})
        predicted_belief = self.T.T @ self.belief

        # 2. 更新步驟
        # P(z_t|x_t) = Z * z_t_proba
        sensor_likelihood = self.Z @ z_t_proba

        # bel(x_t) = η * P(z_t | x_t) * bel'(x_t)
        updated_belief_ = sensor_likelihood * predicted_belief

        # Normalize (η)
        norm_factor = np.sum(updated_belief_)
        if norm_factor > 1e-9:
            self.belief = updated_belief_ / norm_factor
        else:
            # 如果機率總和趨近於零，重設為均勻分佈以避免錯誤
            self.belief = np.ones_like(self.belief) / len(self.belief)

        return self.belief
