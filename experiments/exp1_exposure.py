#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验1：曝光时间对成像的影响（静止 / 运动）"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from camera.mvs_camera import MvsCamera

ROOT = '/home/jie/桌面/yolo-armorinference'
OUT = os.path.join(ROOT, 'data/exp/exp1')


def img_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return dict(
        mean=round(float(img.mean()), 2),
        p99=round(float(np.percentile(g, 99)), 1),
        max=int(g.max()),
        saturated_pct=round(float((g >= 250).mean() * 100), 3),
        sharpness=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exposures', default='500,1000,3000')
    ap.add_argument('--gain', type=float, default=0.0)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--tag', default='static')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    exposures = [float(x) for x in args.exposures.split(',')]
    results = []
    with MvsCamera() as cam:
        print('camera:', cam.info)
        cam.set_pixel_format('BayerRG8')
        cam.set_gain(args.gain)
        cam.set_frame_rate(30)
        cam.start()
        for us in exposures:
            cam.set_exposure(us)
            time.sleep(0.4)
            got = 0
            for i in range(args.repeats):
                img = None
                for _ in range(6):
                    img = cam.grab(1000)
                    if img is not None:
                        break
                if img is None:
                    print('grab failed at %.0f us' % us)
                    continue
                fn = 'exp1_%s_%dus_%d.png' % (args.tag, int(us), i)
                cv2.imwrite(os.path.join(OUT, fn), img)
                s = img_stats(img)
                s.update(exposure_us=us, gain_db=args.gain, file=fn)
                results.append(s)
                got += 1
                print("  %6d us #%d  mean=%6.2f  sat=%5.3f%%  sharp=%8.1f" %
                      (us, i, s['mean'], s['saturated_pct'], s['sharpness']))
            if got == 0:
                print('!!! no frame at', us)
        cam.stop()
    with open(os.path.join(OUT, 'exp1_result_%s.json' % args.tag), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
