#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机采集 → 装甲板识别 → 角点提取 → PnP 位姿解算 → 卡尔曼滤波 → 运动预测

用法::

    python experiments/exp7_pnp_kalman.py --calib calibration/calib_result.json
    python experiments/exp7_pnp_kalman.py --calib ... --record data/exp/exp7_demo.mp4
    python experiments/exp7_pnp_kalman.py --calib ... --debug-order

按键：q 或 ESC 退出
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from camera.mvs_camera import MvsCamera  # noqa: E402
from pose.armor_pnp import (  # noqa: E402
    DEFAULT_ARMOR_H_MM, DEFAULT_ARMOR_W_MM,
    draw_axes, reprojection_error, solve_pose)
from pose.kalman_tracker import KalmanTracker  # noqa: E402
from utils import imwrite_u  # noqa: E402


def load_calib(path):
    if path.lower().endswith('.json'):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return (np.array(d['camera_matrix'], dtype=np.float64),
                np.array(d['dist_coeffs'], dtype=np.float64),
                int(d['image_width']), int(d['image_height']),
                float(d.get('rms', d.get('rms_fix_k3', d.get('rms_default5', 0.0)))))
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit('打不开标定文件：%s' % path)
    K = fs.getNode('camera_matrix').mat()
    dist = fs.getNode('dist_coeffs').mat()
    w = int(fs.getNode('image_width').real())
    h = int(fs.getNode('image_height').real())
    rms = float(fs.getNode('rms').real())
    fs.release()
    if K is None:
        raise SystemExit('标定文件里没有 camera_matrix')
    return K, dist, w, h, rms


def pick_target(result):
    """取置信度最高的检测框，返回 (xyxy, conf, cls, kpts) 或 None"""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    idx = int(result.boxes.conf.argmax().item())
    xyxy = result.boxes.xyxy[idx].cpu().numpy()
    conf = float(result.boxes.conf[idx].item())
    cls_id = int(result.boxes.cls[idx].item())
    kpts = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        kpts = result.keypoints.xy[idx].cpu().numpy()
    return xyxy, conf, cls_id, kpts


def draw_debug_order(frame, kpts, names):
    """调试用：给每个关键点画编号，确认模型输出的顺序"""
    vis = frame.copy()
    for i, (x, y) in enumerate(kpts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(vis, p, 6, (0, 255, 255), -1)
        label = '%d:%s' % (i, names[i] if i < len(names) else '?')
        cv2.putText(vis, label, (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', default='calibration/calib_result.yaml')
    ap.add_argument('--weights', default='weights/best.pt')
    ap.add_argument('--armor-w', type=float, default=DEFAULT_ARMOR_W_MM)
    ap.add_argument('--armor-h', type=float, default=DEFAULT_ARMOR_H_MM)
    ap.add_argument('--exposure', type=float, default=10000)
    ap.add_argument('--gain', type=float, default=0)
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--record', default='', help='录制视频路径；留空不录')
    ap.add_argument('--log', default='data/exp/exp7_pnp_log.json')
    ap.add_argument('--duration', type=float, default=0, help='运行秒数，0 表示手动退出')
    ap.add_argument('--no-window', action='store_true', help='无界面模式')
    ap.add_argument('--warmup', type=float, default=0, help='开始计时前等待的秒数，给操作者留缓冲')
    ap.add_argument('--debug-order', action='store_true', help='只显示关键点编号并保存调试图')
    args = ap.parse_args()

    K, dist, cw, ch, rms = load_calib(args.calib)
    print('标定：%dx%d  RMS=%.4f px' % (cw, ch, rms))
    print('内参：\n%s' % K)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    cls_names = model.names
    print('模型类别：%s' % cls_names)

    tracker = KalmanTracker(dt=1.0 / 30.0)
    records = []
    writer = None
    names = ['TL', 'BL', 'BR', 'TR']

    with MvsCamera() as cam:
        print('相机：%s 序列号 %s' % (cam.info.get('model'), cam.info.get('serial')))
        cam.set_pixel_format('BayerRG8')
        cam.set_exposure(args.exposure)
        cam.set_gain(args.gain)
        cam.start()

        if args.warmup > 0:
            print('预热 %.0f 秒，之后开始记录...' % args.warmup, flush=True)
            time.sleep(args.warmup)
        t0 = time.time()
        n_frames = 0
        win = 'exp7 pnp + kalman'
        if not args.no_window:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 960)

        last_t = None
        while True:
            frame = cam.grab()
            if frame is None:
                continue
            n_frames += 1
            now = time.time()
            if last_t is not None:
                tracker.dt = max(1e-3, now - last_t)
            last_t = now

            res = model(frame, conf=args.conf, verbose=False)
            vis = frame.copy()
            target = pick_target(res[0]) if res else None

            line = None
            if target is not None:
                xyxy, conf, cls_id, kpts = target
                if kpts is not None and len(kpts) >= 4:
                    if args.debug_order:
                        vis = draw_debug_order(vis, kpts, names)
                        if n_frames == 5:
                            imwrite_u('data/exp/exp7_order_debug.png', vis)
                    else:
                        corners = np.asarray(kpts[:4], dtype=np.float32)
                        rvec, tvec, err = solve_pose(
                            corners, K, dist, args.armor_w, args.armor_h)
                        if rvec is not None:
                            tvec_r = tvec.ravel()
                            pos_f, vel_f = tracker.update(tvec_r)
                            pred50 = tracker.predict_ahead(0.05)
                            dist_mm = float(np.linalg.norm(tvec_r))
                            dist_f = float(np.linalg.norm(pos_f))

                            x1, y1, x2, y2 = [int(v) for v in xyxy]
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            for i, (cx, cy) in enumerate(corners):
                                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                                cv2.putText(vis, str(i), (int(cx) + 5, int(cy) - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            draw_axes(vis, rvec, tvec, K, dist, length_mm=0.5 * args.armor_h)

                            name = cls_names.get(cls_id, str(cls_id)) if isinstance(cls_names, dict) else str(cls_id)
                            lines = [
                                '%s conf=%.2f reproj=%.2fpx' % (name, conf, err),
                                'XYZ = (%7.1f, %7.1f, %7.1f) mm' % tuple(tvec_r),
                                'filt= (%7.1f, %7.1f, %7.1f) mm' % tuple(pos_f),
                                'dist raw %.1f mm  filt %.1f mm' % (dist_mm, dist_f),
                                'vel = (%6.0f, %6.0f, %6.0f) mm/s' % tuple(vel_f),
                                'pred+50ms = (%7.1f, %7.1f, %7.1f) mm' % tuple(pred50),
                            ]
                            for i, s in enumerate(lines):
                                cv2.putText(vis, s, (20, 40 + i * 34),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                            (0, 255, 0), 2)
                            records.append({
                                't': round(now - t0, 4),
                                'cls': name, 'conf': round(conf, 4),
                                'reproj_px': round(err, 4),
                                'kpts': [[round(float(a), 2), round(float(b), 2)] for a, b in corners],
                                'tvec_raw': [round(float(v), 3) for v in tvec_r],
                                'tvec_filt': [round(float(v), 3) for v in pos_f],
                                'vel': [round(float(v), 2) for v in vel_f],
                                'pred_50ms': [round(float(v), 3) for v in pred50],
                                'dist_mm': round(dist_mm, 2),
                            })
            else:
                cv2.putText(vis, 'no target', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

            if args.record:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'),
                                             25.0, (w, h))
                writer.write(vis)

            if not args.no_window:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            if args.duration > 0 and (now - t0) >= args.duration:
                break

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print('\n处理 %d 帧，用时 %.2f s，%0.2f fps' % (n_frames, elapsed, n_frames / max(elapsed, 1e-6)))
    if records:
        raw = np.array([r['tvec_raw'] for r in records])
        filt = np.array([r['tvec_filt'] for r in records])
        print('原始测量标准差 (x,y,z) = %s' % np.round(raw.std(axis=0), 2))
        print('滤波后标准差 (x,y,z) = %s' % np.round(filt.std(axis=0), 2))
        os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
        with open(args.log, 'w', encoding='utf-8') as f:
            json.dump({'frames': n_frames, 'fps': round(n_frames / max(elapsed, 1e-6), 2),
                       'calib': {'w': cw, 'h': ch, 'rms': rms},
                       'armor_mm': [args.armor_w, args.armor_h],
                       'records': records}, f, ensure_ascii=False, indent=1)
        print('日志已保存 %s' % args.log)


if __name__ == '__main__':
    main()
