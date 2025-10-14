#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class KalmanFilter:
    def __init__(self, x_init, y_init, dt=0.1):
        """
        初始化一個用於 2D 物體追蹤的卡爾曼濾波器。

        :param x_init: 物體的初始 x 座標。
        :param y_init: 物體的初始 y 座標。
        :param dt: 時間間隔 (兩次雷射掃描之間的時間)。
        """
        # 狀態向量 [x, y, vx, vy]' (位置x, 位置y, 速度x, 速度y)
        self.x = np.array([x_init, y_init, 0.0, 0.0]).reshape(4, 1)

        # 狀態轉移矩陣 F (描述物體如何從一個狀態變到下一個狀態)
        # x_new = x_old + vx * dt
        # y_new = y_old + vy * dt
        # vx_new = vx_old
        # vy_new = vy_old
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # 觀測矩陣 H (將狀態向量映射到觀測空間)
        # 我們只能觀測到位置 (x, y)，不能直接觀測到速度
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # 狀態協方差矩陣 P (表示我們對狀態估計的不確定性)
        # 初始時，我們對位置比較確定，但對速度非常不確定
        self.P = np.eye(4) * 10.0
        self.P[0, 0] = self.P[1, 1] = 0.1  # 位置的不確定性較小

        # 過程噪聲協方差矩陣 Q (表示模型預測本身的不確定性，例如突發的加速度)
        self.Q = np.eye(4) * 0.05

        # 觀測噪聲協方差矩陣 R (表示觀測值的不確定性，即分類器的偵測誤差)
        self.R = np.eye(2) * 0.05

    def predict(self, dt=0.1):
        """
        預測步驟：估計物體在下一時刻的狀態。
        """
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # x_pred = F * x
        self.x = self.F @ self.x
        # P_pred = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    def update(self, z):
        """
        更新步驟：使用觀測值 z 來修正預測。

        :param z: 觀測向量 [x, y]'，來自 Adaboost 分類器的偵測結果。
        """
        z = np.array(z).reshape(2, 1)

        # y = z - H * x_pred (觀測殘差)
        y = z - self.H @ self.x

        # S = H * P_pred * H' + R (殘差協方差)
        S = self.H @ self.P @ self.H.T + self.R

        # K = P_pred * H' * S^-1 (卡爾曼增益)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # x_new = x_pred + K * y
        self.x = self.x + K @ y

        # P_new = (I - K * H) * P_pred
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
