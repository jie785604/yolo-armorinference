#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用采集到的棋盘格图像做相机标定

用法::

    python calibration/calibrate.py --images calibration/images --cols 9 --rows 6 --square-mm 25

输出 calibration/calib_result.yaml，包含内参矩阵、畸变系数、重投影误差与图像尺寸。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imread_u  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='calibration/images')
    ap.add_argument('--cols', type=int, default=9, help='内角点列数')
    ap.add_argument('--rows', type=int, default=6, help='内角点行数')
    ap.add_argument('--square-mm', type=float, default=25.0)
    ap.add_argument('--out', default='calibration/calib_result.yaml')
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    # 三维世界坐标：Z=0 平面，X 向右，Y 向下（与图像坐标一致）
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    files = sorted(glob.glob(os.path.join(args.images, '*.png'))
                   + glob.glob(os.path.join(args.images, '*.jpg')))
    if not files:
        raise SystemExit('没有找到标定图像：%s' % args.images)

    obj_points = []
    img_points = []
    used = []
    gray_shape = None
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for f in files:
        img = imread_u(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            print('跳过（未找到角点）：%s' % os.path.basename(f))
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(os.path.basename(f))
        print('使用：%s' % os.path.basename(f))

    if len(used) < 6:
        raise SystemExit('有效图像只有 %d 张，至少需要 6 张' % len(used))

    h, w = gray_shape
    print('\n标定中，图像尺寸 %d x %d，有效样本 %d 张' % (w, h, len(used)))
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    print('重投影误差 RMS = %.4f px' % ret)
    print('内参矩阵 K =')
    print(K)
    print('畸变系数 =', dist.ravel())

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', w)
    fs.write('image_height', h)
    fs.write('board_cols', args.cols)
    fs.write('board_rows', args.rows)
    fs.write('square_mm', args.square_mm)
    fs.write('rms', float(ret))
    fs.write('camera_matrix', K)
    fs.write('dist_coeffs', dist)
    fs.write('used_images', ';'.join(used))
    fs.release()
    print('\n已保存 %s' % args.out)
    if ret > 1.0:
        print('[提示] RMS 偏大（> 1 px），建议重采：避免图像模糊、板子不平、角点不全可见')


if __name__ == '__main__':
    main()
