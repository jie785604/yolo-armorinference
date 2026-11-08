"""实验 1m：运动拖影采集。

让装甲板在相机前横向移动，按不同曝光依次录制，每档保存若干样本帧，
并统计清晰度（Laplacian 方差）与帧间差异（absdiff 均值）作为拖影强度指标。

用法：
    python experiments/exp1m_motion_capture.py \
        --exposures 500,1000,3000,10000,30000 --segment 8
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from camera.mvs_camera import MvsCamera


def main():
    ap = argparse.ArgumentParser(description='多曝光运动拖影采集')
    ap.add_argument('--exposures', default='500,1000,3000,10000,30000',
                    help='逗号分隔的曝光时间列表（μs）')
    ap.add_argument('--segment', type=float, default=8.0, help='每档录制秒数')
    ap.add_argument('--gain', type=float, default=0.0, help='增益 dB')
    ap.add_argument('--samples', type=int, default=8, help='每档保存样本帧数')
    ap.add_argument('--dir', default=os.path.join(ROOT, 'data/exp/exp1m'))
    args = ap.parse_args()

    exposures = [int(x) for x in args.exposures.split(',')]
    os.makedirs(args.dir, exist_ok=True)
    results = []

    with MvsCamera() as cam:
        cam.set_pixel_format('BayerRG8')
        cam.set_gain(args.gain)
        cam.start()
        print('RECORDING NOW (multi-exposure)', flush=True)
        for us in exposures:
            cam.set_exposure(us)
            time.sleep(0.5)
            for _ in range(3):
                cam.grab(500)
            t0 = time.time()
            idx = saved = 0
            prev = None
            shs, dfs = [], []
            while time.time() - t0 < args.segment:
                img = cam.grab(300)
                if img is None:
                    continue
                idx += 1
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                shs.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
                dfs.append(float(cv2.absdiff(g, prev).mean()) if prev is not None else 0.0)
                prev = g
                if idx % 5 == 1 and saved < args.samples:
                    cv2.imwrite(os.path.join(args.dir, 'seq_%05dus_%02d.png' % (us, saved)), img)
                    saved += 1
            r = dict(exposure_us=us, frames=idx, saved=saved,
                     sharp_mean=round(float(np.mean(shs)), 2),
                     diff_mean=round(float(np.mean(dfs)), 2),
                     diff_max=round(float(np.max(dfs)), 2))
            print(r, flush=True)
            results.append(r)

    out = os.path.join(args.dir, 'exp1m_motion_result.json')
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
