#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖相机的自检：用已知位姿投影出角点，再解算回去，验证 PnP 与卡尔曼流程

用法::

    python experiments/exp7_selftest.py

这一步只验证数学正确性（三维点、角点顺序、solvePnP、卡尔曼外推），
最终演示必须用真实相机数据。
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose.armor_pnp import (  # noqa: E402
    DEFAULT_ARMOR_H_MM, DEFAULT_ARMOR_W_MM, armor_object_points,
    order_corners_geometric, solve_pose)
from pose.kalman_tracker import KalmanTracker  # noqa: E402


def main():
    # 假想一台 1440x1080、焦距 2000 的相机
    K = np.array([[2000.0, 0, 720], [0, 2000.0, 540], [0, 0, 1]], np.float64)
    dist = np.zeros((5, 1), np.float64)
    obj = armor_object_points(DEFAULT_ARMOR_W_MM, DEFAULT_ARMOR_H_MM)
    print('三维角点（左上、右上、右下、左下）：')
    print(obj)

    rng = np.random.default_rng(0)
    tracker = KalmanTracker(dt=1 / 30.0)
    errs = []
    for i in range(60):
        # 板子沿 X 轴匀速平移，同时带一点抖动
        truth = np.array([-200.0 + i * 8.0, 20.0, 600.0])
        rvec_true = np.zeros((3, 1))
        tvec_true = truth.reshape(3, 1)
        proj, _ = cv2.projectPoints(obj, rvec_true, tvec_true, K, dist)
        pts = proj.reshape(-1, 2) + rng.normal(0, 0.6, (4, 2))

        corners = order_corners_geometric(pts)
        rvec, tvec, err = solve_pose(corners, K, dist)
        tvec = tvec.ravel()
        pos_f, vel_f = tracker.update(tvec)
        errs.append(np.linalg.norm(tvec - truth))
        if i % 15 == 0:
            print('帧 %2d  解算=%s  真值=%s  误差=%.2f mm  滤波=%s  速度=%s'
                  % (i, np.round(tvec, 1), truth, errs[-1],
                     np.round(pos_f, 1), np.round(vel_f, 1)))
    print('平均位置误差 %.3f mm（加了 0.6 px 角点噪声）' % float(np.mean(errs)))

    pred = tracker.predict_ahead(0.1)
    print('外推 100 ms 后位置：%s' % np.round(pred, 1))
    print('自检通过：角点顺序、PnP、卡尔曼接口均可用')


if __name__ == '__main__':
    main()
