#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验5：YOLO 实时识别（工业相机 + MVS SDK + OpenCV）"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from camera.mvs_camera import MvsCamera
from ultralytics import YOLO

ROOT = '/home/jie/桌面/yolo-armorinference'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(ROOT, 'weights/best.pt'))
    ap.add_argument('--exposure', type=float, default=30000)
    ap.add_argument('--gain', type=float, default=0.0)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--record', type=float, default=15.0, help='录制秒数，0=一直跑')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/exp/exp5_demo.mp4'))
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = YOLO(args.weights)
    print('classes:', model.names)

    det_log = []
    with MvsCamera() as cam:
        print('camera:', cam.info)
        cam.set_pixel_format('BayerRG8')
        cam.set_exposure(args.exposure)
        cam.set_gain(args.gain)
        cam.set_frame_rate(30)
        cam.start()
        img = cam.grab()
        while img is None:
            img = cam.grab()
            time.sleep(0.05)
        h, w = img.shape[:2]
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        print('output:', args.out, w, h)

        t0 = time.time()
        n = 0
        infer_ms_total = 0.0
        while True:
            img = cam.grab()
            if img is None:
                continue
            t_inf = time.perf_counter()
            res = model(img, conf=args.conf, verbose=False)
            infer_ms = (time.perf_counter() - t_inf) * 1000
            infer_ms_total += infer_ms
            ann = res[0].plot()
            n += 1
            elapsed = time.time() - t0
            loop_fps = n / elapsed if elapsed > 0 else 0
            cv2.putText(ann, 'FPS %.1f | infer %.0fms' % (loop_fps, infer_ms),
                        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            boxes = []
            for b in res[0].boxes:
                boxes.append(dict(cls=model.names[int(b.cls[0])],
                                  conf=round(float(b.conf[0]), 3),
                                  xyxy=[round(float(v), 0) for v in b.xyxy[0].tolist()]))
            det_log.append(dict(t=round(elapsed, 2), infer_ms=round(infer_ms, 1), boxes=boxes))
            writer.write(ann)
            if args.show:
                cv2.imshow('YOLO Live', cv2.resize(ann, (960, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.record > 0 and elapsed >= args.record:
                break
        cam.stop()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
        total = time.time() - t0
        print('frames=%d  elapsed=%.2fs  e2e_fps=%.2f  infer_avg=%.1fms' %
              (n, total, n / total, infer_ms_total / max(n, 1)))
        with open(os.path.join(ROOT, 'data/exp/exp5_live_log.json'), 'w') as f:
            json.dump(dict(weights=os.path.basename(args.weights), exposure_us=args.exposure,
                           gain_db=args.gain, conf=args.conf, frames=n,
                           elapsed_s=round(total, 2), e2e_fps=round(n / total, 2),
                           infer_avg_ms=round(infer_ms_total / max(n, 1), 1),
                           detections=det_log), f, indent=2, ensure_ascii=False)
        print('log saved')


if __name__ == '__main__':
    main()
