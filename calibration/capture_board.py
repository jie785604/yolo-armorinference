#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集棋盘格标定图像

用法::

    python calibration/capture_board.py --out calibration/images --cols 9 --rows 6 --square-mm 25

操作：
    空格  检测到角点后保存一张
    q/ESC 退出

建议采集 15~25 张，覆盖：
    不同距离（近 / 中 / 远）、不同角度（左右倾斜、上下倾斜、旋转），
    板子放在画面不同区域（中心、四角），角点尽量清晰。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.mvs_camera import MvsCamera  # noqa: E402
from utils import imwrite_u  # noqa: E402


def find_corners(gray, pattern):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='calibration/images')
    ap.add_argument('--cols', type=int, default=9)
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--exposure', type=float, default=10000)
    ap.add_argument('--gain', type=float, default=0)
    ap.add_argument('--no-window', action='store_true', help='无界面模式下自动保存')
    ap.add_argument('--auto-count', type=int, default=0, help='无界面模式保存张数')
    ap.add_argument('--auto-interval', type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pattern = (args.cols, args.rows)
    idx = len([f for f in os.listdir(args.out) if f.endswith('.png')])

    with MvsCamera() as cam:
        print('相机：%s  序列号 %s' % (cam.info.get('model'), cam.info.get('serial')))
        cam.set_pixel_format('BayerRG8')
        cam.set_exposure(args.exposure)
        cam.set_gain(args.gain)
        cam.start()

        if args.no_window:
            n = 0
            last = 0.0
            while n < args.auto_count:
                frame = cam.grab()
                if frame is None:
                    continue
                now = time.time()
                if now - last < args.auto_interval:
                    continue
                last = now
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, corners = find_corners(gray, pattern)
                if not ok:
                    print('  未检测到角点，跳过')
                    continue
                path = os.path.join(args.out, 'board_%02d.png' % idx)
                imwrite_u(path, frame)
                print('  保存 %s（角点 %d 个）' % (path, len(corners)))
                idx += 1
                n += 1
        else:
            win = 'capture board'
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            while True:
                frame = cam.grab()
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, corners = find_corners(gray, pattern)
                vis = frame.copy()
                if ok:
                    cv2.drawChessboardCorners(vis, pattern, corners, ok)
                cv2.putText(vis, 'found' if ok else 'searching', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0) if ok else (0, 0, 255), 3)
                cv2.putText(vis, 'saved: %d   SPACE=save  q=quit' % idx, (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    if not ok:
                        print('未检测到角点，未保存')
                        continue
                    path = os.path.join(args.out, 'board_%02d.png' % idx)
                    imwrite_u(path, frame)
                    print('保存 %s（角点 %d 个）' % (path, len(corners)))
                    idx += 1
                elif key in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()

    print('共 %d 张，保存在 %s' % (idx, args.out))


if __name__ == '__main__':
    main()
