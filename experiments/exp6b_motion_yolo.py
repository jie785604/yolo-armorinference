"""实验 6b：运动模糊对 YOLO 识别的影响。

对 exp1m 各曝光档的运动帧做推理，统计：
  - 装甲板可见帧（红色 LED 像素足够多）的检出率
  - 平均检出框数、R1 平均置信度
"""
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def led_visible(img, min_px=500):
    """用红色 LED 亮斑判断装甲板是否在画面内。"""
    b, g, r = cv2.split(img)
    mask = (r > 200) & (g < 100)
    return int(mask.sum()) >= min_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(ROOT, 'weights/best.pt'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--dir', default=os.path.join(ROOT, 'data/exp/exp1m'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/exp/exp6b_motion_yolo.json'))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, 'seq_*us_*.png')))
    groups = {}
    for f in files:
        m = re.search(r'seq_(\d+)us_', os.path.basename(f))
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)

    model = YOLO(args.weights)
    report = {'conf': args.conf, 'weights': args.weights, 'groups': {}}

    print('%-8s %-6s %-8s %-9s %-9s %-10s' % ('exposure', 'frames', 'visible', 'det_rate', 'avg_boxes', 'r1_conf'))
    for us in sorted(groups):
        fs = groups[us]
        vis = 0
        hit = 0
        boxes_total = 0
        r1_confs = []
        for f in fs:
            img = cv2.imread(f)
            if img is None:
                continue
            if not led_visible(img):
                continue
            vis += 1
            res = model(img, conf=args.conf, verbose=False)
            boxes = res[0].boxes
            boxes_total += len(boxes)
            names = res[0].names
            got_r1 = False
            for b in boxes:
                cls = names[int(b.cls)]
                cf = float(b.conf)
                if cls == 'R1':
                    r1_confs.append(cf)
                    got_r1 = True
            if got_r1:
                hit += 1
        rate = hit / vis if vis else 0.0
        avg_boxes = boxes_total / vis if vis else 0.0
        avg_conf = float(np.mean(r1_confs)) if r1_confs else 0.0
        print('%-8d %-6d %-8d %-9.2f %-9.2f %-10.3f' % (us, len(fs), vis, rate, avg_boxes, avg_conf))
        report['groups'][str(us)] = {
            'frames': len(fs), 'visible': vis, 'r1_detected': hit,
            'r1_detect_rate': round(rate, 4),
            'avg_boxes': round(avg_boxes, 3),
            'r1_avg_conf': round(avg_conf, 4),
        }

    with open(args.out, 'w') as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print('saved', args.out)


if __name__ == '__main__':
    main()
