#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验6：相机参数（曝光/增益）对 YOLO 识别结果的影响"""
import sys, os, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = '/home/jie/桌面/yolo-armorinference'
DATA = os.path.join(ROOT, 'data/exp')


def analyze(model, path, conf):
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = model(img, conf=conf, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        boxes.append(dict(cls=model.names[int(b.cls[0])],
                          conf=round(float(b.conf[0]), 3),
                          area=round(float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])), 0),
                          xyxy=[round(float(v), 0) for v in b.xyxy[0].tolist()]))
    best = max(boxes, key=lambda b: b['conf']) if boxes else None
    return dict(file=os.path.basename(path), mean=round(float(g.mean()), 2),
                saturated_pct=round(float((g >= 250).mean() * 100), 3),
                n_boxes=len(boxes), best=best, boxes=boxes)


def summarize(results, key):
    print('\n--- %s ---' % key)
    print('%-28s %8s %8s %6s %10s %8s' % ('file', 'mean', 'sat%', 'boxes', 'top_cls', 'conf'))
    for r in results:
        b = r['best']
        print('%-28s %8.2f %8.3f %6d %10s %8.3f' % (
            r['file'], r['mean'], r['saturated_pct'], r['n_boxes'],
            b['cls'] if b else '-', b['conf'] if b else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()
    model = YOLO(os.path.join(ROOT, 'weights/best.pt'))

    out = {}
    for tag, pattern in [('exp1_exposure', os.path.join(DATA, 'exp1', 'exp1_static_*.png')),
                         ('exp2_gain', os.path.join(DATA, 'exp2', 'exp2_gain*.png'))]:
        files = sorted(glob.glob(pattern))
        results = [analyze(model, f, args.conf) for f in files]
        out[tag] = results
        summarize(results, tag)

    with open(os.path.join(DATA, 'exp6_result.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\nsaved exp6_result.json')


if __name__ == '__main__':
    main()
