#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""匀速模型卡尔曼滤波，用于装甲板三维位置降抖与短时预测

状态量 x = [x, y, z, vx, vy, vz]^T（毫米、毫米/秒）
观测量 z = [x, y, z]^T，来自 PnP 解算的 tvec
"""
import cv2
import numpy as np


class KalmanTracker:
    def __init__(self, dt=1.0 / 30.0,
                 process_var=20.0,
                 measure_var=100.0):
        self.kf = cv2.KalmanFilter(6, 3)
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = F
        self.kf.measurementMatrix = H
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_var
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measure_var
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * measure_var
        self.dt = dt
        self.initialized = False

    def reset(self):
        self.initialized = False

    def update(self, z):
        """输入测量值 [x,y,z]，返回滤波后的位置与速度"""
        z = np.asarray(z, dtype=np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost = np.array(
                [[z[0, 0]], [z[1, 0]], [z[2, 0]], [0], [0], [0]], np.float32)
            self.initialized = True
            return self.position(), self.velocity()
        self.kf.predict()
        self.kf.correct(z)
        return self.position(), self.velocity()

    def position(self):
        return self.kf.statePost[:3].ravel().astype(np.float64)

    def velocity(self):
        return self.kf.statePost[3:].ravel().astype(np.float64)

    def predict_ahead(self, seconds):
        """按当前速度外推若干秒后的位置（毫米）"""
        return self.position() + self.velocity() * float(seconds)
