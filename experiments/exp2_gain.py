#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验2：增益对图像质量的影响（0 / 6 / 12 dB）"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from camera.mvs_camera import MvsCamera

ROOT = '/home/jie/桌面/yolo-armorinference'
OUT = os.path.join(ROOT, 'data/exp/exp2')


def img_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    # 暗部区域：画面中亮度最低的 25% 像素
    thr = np.percentile(g, 25)
    dark = g[g <= thr]
    return dict(
        mean=round(float(img.mean()), 2),
        p99=round(float(np.percentile(g, 99)), 1),
        max=int(g.max()),
        saturated_pct=round(float((g >= 250).mean() * 100), 3),
        dark_mean=round(float(dark.mean()), 2),
        dark_noise_std=round(float(dark.std()), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gains', default='0,6,12')
    ap.add_argument('--exposure', type=float, default=3000)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    gains = [float(x) for x in args.gains.split(',')]
    results = []
    with MvsCamera() as cam:
        print('camera:', cam.info)
        cam.set_pixel_format('BayerRG8')
        cam.set_exposure(args.exposure)
        cam.set_frame_rate(30)
        cam.start()
        for gdb in gains:
            cam.set_gain(gdb)
            time.sleep(0.4)
            for i in range(args.repeats):
                img = None
                for _ in range(6):
                    img = cam.grab(1000)
                    if img is not None:
                        break
                if img is None:
                    print('grab failed at %.1f dB' % gdb)
                    continue
                fn = 'exp2_gain%d_%d.png' % (int(gdb), i)
                cv2.imwrite(os.path.join(OUT, fn), img)
                s = img_stats(img)
                s.update(gain_db=gdb, exposure_us=args.exposure, file=fn)
                results.append(s)
                print("  gain=%5.1f dB #%d  mean=%6.2f  sat=%5.3f%%  dark_noise=%5.2f" %
                      (gdb, i, s['mean'], s['saturated_pct'], s['dark_noise_std']))
        cam.stop()
    with open(os.path.join(OUT, 'exp2_result.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
