#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容中文路径的图像读写（Windows 下 cv2.imread/imwrite 不支持非 ASCII 路径）"""
import os

import cv2
import numpy as np


def imread_u(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path, img, params=None):
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    buf.tofile(path)
    return True
