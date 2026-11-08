#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验4b：OpenCV/NumPy 图像内存行为（对应课堂 mat_memory 示例）

演示三件事：
1. 浅拷贝（视图）不占新内存，深拷贝才真正分配
2. 持有大量图像引用会让进程内存持续增长
3. 释放引用后内存可以回收
"""
import gc
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, W = 1080, 1440


def rss_mb():
    with open('/proc/self/status') as fp:
        for line in fp:
            if line.startswith('VmRSS'):
                return round(int(line.split()[1]) / 1024.0, 1)
    return 0.0


def main():
    out = {'image_size_mb': round(H * W * 3 / 1024.0 / 1024.0, 2), 'steps': []}

    def step(name, **kw):
        rec = {'step': name, 'rss_mb': rss_mb()}
        rec.update(kw)
        out['steps'].append(rec)
        print('%-34s rss=%7.1f MB' % (name, rec['rss_mb']), flush=True)

    step('start')

    img = np.zeros((H, W, 3), np.uint8)
    step('allocate 1 frame (1440x1080x3)')

    views = [img] * 500
    step('500 shallow refs (list of views)', n=len(views))

    copies = [img.copy() for _ in range(20)]
    step('20 deep copies', n=len(copies), expect_mb=round(20 * H * W * 3 / 1024.0 / 1024.0, 1))

    conv = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _ in range(20)]
    step('20 cvtColor results (gray)', n=len(conv))

    del copies, conv
    gc.collect()
    step('after del copies/conv + gc')

    leak = []
    for _ in range(100):
        leak.append(np.zeros((H, W, 3), np.uint8))
    step('hold 100 frames (simulated leak)', n=len(leak))

    del leak
    gc.collect()
    step('after releasing 100 frames')

    base = rss_mb()
    frames = []
    for i in range(300):
        frames.append(np.zeros((H, W, 3), np.uint8))
        if len(frames) > 5:
            frames.pop(0)
    step('300-iteration ring buffer of 5 frames', delta_mb=round(rss_mb() - base, 1))

    os.makedirs(os.path.join(ROOT, 'data/exp'), exist_ok=True)
    path = os.path.join(ROOT, 'data/exp/exp4b_mat_memory.json')
    with open(path, 'w') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print('saved', path)


if __name__ == '__main__':
    main()
