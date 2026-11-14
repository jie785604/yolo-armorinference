#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成可打印的棋盘格标定板图片（A4，300 dpi）

用法::

    python calibration/make_board.py --cols 9 --rows 6 --square-mm 25

生成的 PNG 按 300 dpi 打印时，每格边长正好等于 --square-mm 毫米。
打印时务必选择“实际大小 / 100%”，不要选“适合页面”，否则尺寸会变。
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imwrite_u  # noqa: E402

MM_PER_INCH = 25.4
DPI = 300


def build_board(cols, rows, square_mm, margin_mm=10):
    """生成棋盘格图像，返回 (img, 每格像素数)"""
    px_per_mm = DPI / MM_PER_INCH
    sq = int(round(square_mm * px_per_mm))
    margin = int(round(margin_mm * px_per_mm))
    # 棋盘格需要 (cols+1) x (rows+1) 个格子
    board_w = sq * (cols + 1)
    board_h = sq * (rows + 1)
    img = np.full((board_h + 2 * margin, board_w + 2 * margin), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = margin + r * sq
                x0 = margin + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 0
    return img, sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cols', type=int, default=9, help='内角点列数')
    ap.add_argument('--rows', type=int, default=6, help='内角点行数')
    ap.add_argument('--square-mm', type=float, default=25.0, help='每格边长（毫米）')
    ap.add_argument('--out', default='calibration/board_9x6_25mm.png')
    args = ap.parse_args()

    img, sq = build_board(args.cols, args.rows, args.square_mm)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    imwrite_u(args.out, img)
    h, w = img.shape[:2]
    print('已生成 %s' % args.out)
    print('  内角点 %d x %d，每格 %.1f mm，单格 %d px' % (args.cols, args.rows, args.square_mm, sq))
    print('  图像 %d x %d px，按 300 dpi 打印为 %.1f x %.1f mm（A4 是 210 x 297 mm）'
          % (w, h, w / DPI * MM_PER_INCH, h / DPI * MM_PER_INCH))
    if w / DPI * MM_PER_INCH > 210 or h / DPI * MM_PER_INCH > 297:
        print('  [警告] 超出 A4 范围，请减小格子尺寸或内角点数')


if __name__ == '__main__':
    main()
