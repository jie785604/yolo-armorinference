#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验3：图像格式与取流性能对比 (BayerRG8 / BGR8_Planar / RGB8_Planar)"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from camera.mvs_camera import MvsCamera

CLK = os.sysconf('SC_CLK_TCK')

def proc_cpu_seconds():
    with open('/proc/self/stat') as f:
        parts = f.read().split()
    return (int(parts[13]) + int(parts[14])) / CLK

def measure(cam, fmt, duration=6.0):
    cam.set_pixel_format(fmt)
    cam.start()
    for _ in range(5):
        cam.grab()
    t0 = time.time(); c0 = proc_cpu_seconds()
    n = 0
    while time.time() - t0 < duration:
        if cam.grab() is not None:
            n += 1
    dt = time.time() - t0
    cpu = (proc_cpu_seconds() - c0) / dt * 100
    raw = cam.grab(convert=False)
    cam.stop()
    arr, w, h, pf = raw
    t = time.time()
    for _ in range(100):
        cam._decode(arr, w, h, pf)
    decode_ms = (time.time() - t) / 100 * 1000
    return dict(fmt=fmt, fps=round(n / dt, 2), frames=n, seconds=round(dt, 2),
                cpu_percent=round(cpu, 1), decode_ms=round(decode_ms, 2),
                raw_bytes=int(arr.nbytes), w=w, h=h)

def main():
    out = []
    with MvsCamera() as cam:
        cam.set_exposure(3000); cam.set_gain(0); cam.set_frame_rate(30)
        for fmt in ['BayerRG8', 'BGR8_Planar', 'RGB8_Planar']:
            r = measure(cam, fmt)
            out.append(r)
            print(f"{r['fmt']:14s} fps={r['fps']:6.2f} decode={r['decode_ms']:5.2f}ms cpu={r['cpu_percent']:5.1f}% raw={r['raw_bytes']/1e6:.2f}MB")
    os.makedirs('/home/jie/桌面/yolo-armorinference/data/exp', exist_ok=True)
    with open('/home/jie/桌面/yolo-armorinference/data/exp/exp3_format_compare.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved exp3_format_compare.json')

if __name__ == '__main__':
    main()
