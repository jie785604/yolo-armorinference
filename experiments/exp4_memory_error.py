#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验4：内存管理与程序异常退出（取流超时 / 错误图像格式 / 缓冲区归还）"""
import sys, os, time, json, logging
from ctypes import pointer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
from camera.mvs_camera import MvsCamera, CameraError, PIXEL_FORMATS
from MvCameraControl_class import MV_FRAME_OUT, MvCamera

ROOT = '/home/jie/桌面/yolo-armorinference'
OUT = os.path.join(ROOT, 'data/exp')
LOG = os.path.join(OUT, 'exp4_error.log')
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(filename=LOG, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', filemode='w')
log = logging.getLogger('exp4')


def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return -1.0


def fd_count():
    try:
        return len(os.listdir('/proc/self/fd'))
    except Exception:
        return -1


def case_normal(cam):
    log.info('CASE normal: start, grab 100 frames, free each buffer')
    cam.set_pixel_format('BayerRG8')
    cam.start()
    r0, f0 = rss_mb(), fd_count()
    got = 0
    for _ in range(100):
        if cam.grab(1000) is not None:
            got += 1
    r1, f1 = rss_mb(), fd_count()
    cam.stop()
    d = dict(case='normal', grabbed=got, rss_before=r0, rss_after=r1,
             rss_delta=round(r1 - r0, 1), fd_before=f0, fd_after=f1, crashed=False)
    log.info('CASE normal result: %s', d)
    return d


def case_timeout(cam):
    log.info('CASE timeout: grab without StartGrabbing')
    cam.stop()
    t0 = time.time()
    img = cam.grab(500)
    wait_ms = (time.time() - t0) * 1000
    d = dict(case='timeout', returned_none=(img is None),
             wait_ms=round(wait_ms, 1), crashed=False)
    log.info('CASE timeout result: %s', d)
    return d


def case_bad_format(cam):
    log.info('CASE bad_format: set unsupported PixelFormat')
    d = dict(case='bad_format')
    try:
        cam.set_pixel_format('BGR8_Packed')
        d['alias_raised'] = False
    except CameraError as e:
        d['alias_raised'] = True
        d['alias_error'] = str(e)
        log.info('  expected CameraError: %s', e)
    bad = 0xDEADBEEF
    ret = cam.cam.MV_CC_SetEnumValue('PixelFormat', bad)
    d['sdk_ret'] = '0x%08x' % ret
    d['sdk_rejected'] = (ret != 0)
    log.info('  SDK SetEnumValue(0x%08x) -> 0x%08x', bad, ret)
    cam.set_pixel_format('BayerRG8')
    log.info('CASE bad_format recovered to BayerRG8')
    return d


def case_buffer_leak(cam):
    log.info('CASE buffer_leak: never free image buffers, expect exhaustion')
    cam.set_pixel_format('BayerRG8')
    cam.start()
    st = MV_FRAME_OUT()
    leaked = 0
    for i in range(12):
        ret = cam.cam.MV_CC_GetImageBuffer(st, 800)
        if ret == 0:
            leaked += 1
        else:
            log.info('  exhausted at iteration %d ret=0x%08x', i, ret)
            break
    t0 = time.time()
    st2 = MV_FRAME_OUT()
    ret2 = cam.cam.MV_CC_GetImageBuffer(st2, 800)
    wait_ms = (time.time() - t0) * 1000
    if ret2 == 0:
        try:
            cam.cam.MV_CC_FreeImageBuffer(st2)
        except Exception:
            pass
    cam.stop()
    log.info('  leaked buffers=%d, extra grab ret=0x%08x wait=%.0fms', leaked, ret2, wait_ms)
    cam.stop()
    cam.start()
    ok = False
    for _ in range(5):
        if cam.grab(1000) is not None:
            ok = True
            break
    cam.stop()
    d = dict(case='buffer_leak', leaked=leaked, extra_grab_ret='0x%08x' % ret2,
             extra_wait_ms=round(wait_ms, 1), recovered=ok)
    log.info('CASE buffer_leak result: %s', d)
    return d


def case_exception_exit():
    log.info('CASE exception_exit: raise inside with-block, verify cleanup')
    d = dict(case='exception_exit')
    try:
        with MvsCamera() as cam:
            cam.set_pixel_format('BayerRG8')
            cam.start()
            _ = cam.grab(1000)
            raise RuntimeError('simulated pipeline failure')
    except RuntimeError as e:
        d['exception_caught'] = True
        d['message'] = str(e)
        log.info('  caught: %s', e)
    ok = False
    try:
        with MvsCamera() as cam2:
            cam2.set_pixel_format('BayerRG8')
            ok = True
    except Exception as ex:
        d['reopen_error'] = str(ex)
        log.error('  reopen failed: %s', ex)
    d['reopen_ok'] = ok
    d['fd_after'] = fd_count()
    log.info('CASE exception_exit result: %s', d)
    return d


def main():
    log.info('=== experiment 4 start, pid=%d ===', os.getpid())
    results = {}
    try:
        with MvsCamera() as cam:
            log.info('camera opened: %s', cam.info)
            results['normal'] = case_normal(cam)
            results['timeout'] = case_timeout(cam)
            results['bad_format'] = case_bad_format(cam)
            results['buffer_leak'] = case_buffer_leak(cam)
        results['exception_exit'] = case_exception_exit()
        results['final_rss_mb'] = rss_mb()
        results['final_fd'] = fd_count()
        results['exit_ok'] = True
    except Exception as e:
        log.exception('unexpected failure: %s', e)
        results['unexpected_error'] = repr(e)
        results['exit_ok'] = False
    with open(os.path.join(OUT, 'exp4_result.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info('=== experiment 4 end ===')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
