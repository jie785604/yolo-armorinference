#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""海康 MV-CA016-10UC 工业相机封装（MVS SDK + OpenCV）

支持 Windows 与 Linux 两套 MVS 安装路径，运行时自动探测。
"""
import os
import sys
from ctypes import cast, POINTER

import numpy as np
import cv2

# (Python 样例目录, 动态库目录, 是否需要在库目录下拼 /64/)
_SDK_CANDIDATES = [
    (r'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport',
     r'C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64', False),
    (r'C:\Program Files\MVS\Development\Samples\Python\MvImport',
     r'C:\Program Files\Common Files\MVS\Runtime\Win64_x64', False),
    (r'D:\MVS\Development\Samples\Python\MvImport',
     r'D:\MVS\Runtime\Win64_x64', False),
    ('/opt/MVS/Samples/64/Python/MvImport', '/opt/MVS/lib', True),
    ('/opt/MVS/Samples/32/Python/MvImport', '/opt/MVS/lib/32', True),
]


def _setup_sdk():
    """定位 MVS 的 Python 样例目录与动态库目录，返回 (py_dir, lib_dir)"""
    for py_dir, lib_dir, needs_arch in _SDK_CANDIDATES:
        if not os.path.isdir(py_dir):
            continue
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)
        if os.path.isdir(lib_dir):
            os.environ['MVCAM_COMMON_RUNENV'] = lib_dir
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(lib_dir)
                except OSError:
                    pass
            os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
        return py_dir, lib_dir
    raise ImportError(
        '找不到 MVS SDK。Windows 请安装 MVS 客户端，'
        'Linux 请确认 /opt/MVS 存在。已尝试：%s' % [c[0] for c in _SDK_CANDIDATES]
    )


_SDK_PY_DIR, _SDK_LIB_DIR = _setup_sdk()

from MvCameraControl_class import *  # noqa: E402,F403

PIXEL_FORMATS = {
    'Mono8': 0x01080001,
    'BayerRG8': 0x01080009,
    'RGB8_Planar': 0x02180014,
    'BGR8_Planar': 0x02180015,
    'YUV422_YUYV': 0x02100032,
}


class CameraError(RuntimeError):
    pass


class MvsCamera:
    """相机上下文管理器

    用法::

        with MvsCamera() as cam:
            cam.set_exposure(1000)
            cam.start()
            frame = cam.grab()      # BGR ndarray
    """

    def __init__(self, index=0):
        self.cam = None
        self.index = index
        self.info = {}
        self.cur_format = None

    # ---------------- 生命周期 ----------------
    def open(self):
        # 4.x 以上 SDK 才有 Initialize/Finalize，3.x 直接枚举即可
        if hasattr(MvCamera, 'MV_CC_Initialize'):
            MvCamera.MV_CC_Initialize()
        dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, dev_list)
        if ret != 0:
            raise CameraError('枚举设备失败 ret=0x%08x' % ret)
        if dev_list.nDeviceNum == 0:
            raise CameraError('未找到相机（检查 USB 是否插好、是否被 MVS 客户端占用）')
        st_dev = cast(dev_list.pDeviceInfo[self.index], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(st_dev)
        if ret != 0:
            raise CameraError('创建句柄失败 ret=0x%08x' % ret)
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            raise CameraError('打开设备失败 ret=0x%08x（是否被 MVS 客户端占用？）' % ret)
        self.cam.MV_CC_SetEnumValue('TriggerMode', 0)
        self._read_info()
        self.cur_format = self._get_enum('PixelFormat')
        return self

    def close(self):
        if self.cam is not None:
            for fn in (self.cam.MV_CC_StopGrabbing, self.cam.MV_CC_CloseDevice,
                       self.cam.MV_CC_DestroyHandle):
                try:
                    fn()
                except Exception:
                    pass
            self.cam = None
        try:
            if hasattr(MvCamera, 'MV_CC_Finalize'):
                MvCamera.MV_CC_Finalize()
        except Exception:
            pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _ok(ret, what):
        if ret != 0:
            raise CameraError('%s 失败 ret=0x%08x' % (what, ret))

    def _get_str(self, key):
        v = MVCC_STRINGVALUE()
        if self.cam.MV_CC_GetStringValue(key, v) != 0:
            return ''
        return bytes(v.chCurValue).split(b'\x00')[0].decode('utf-8', 'ignore')

    def _get_enum(self, key):
        v = MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue(key, v) != 0:
            return None
        return v.nCurValue

    def _read_info(self):
        self.info = {
            'model': self._get_str('DeviceModelName'),
            'serial': self._get_str('DeviceSerialNumber'),
            'version': self._get_str('DeviceVersion'),
            'firmware': self._get_str('DeviceFirmwareVersion'),
        }

    # ---------------- 参数 ----------------
    def set_exposure(self, us):
        self._ok(self.cam.MV_CC_SetEnumValue('ExposureAuto', 0), 'ExposureAuto=Off')
        self._ok(self.cam.MV_CC_SetFloatValue('ExposureTime', float(us)),
                 'ExposureTime=%.1f' % us)

    def set_gain(self, db):
        self._ok(self.cam.MV_CC_SetEnumValue('GainAuto', 0), 'GainAuto=Off')
        self._ok(self.cam.MV_CC_SetFloatValue('Gain', float(db)), 'Gain=%.2f' % db)

    def set_frame_rate(self, fps):
        self.cam.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', True)
        self._ok(self.cam.MV_CC_SetFloatValue('AcquisitionFrameRate', float(fps)),
                 'AcquisitionFrameRate=%.1f' % fps)

    def set_roi(self, offset_x, offset_y, width, height):
        self._ok(self.cam.MV_CC_SetIntValue('OffsetX', int(offset_x)), 'OffsetX')
        self._ok(self.cam.MV_CC_SetIntValue('OffsetY', int(offset_y)), 'OffsetY')
        self._ok(self.cam.MV_CC_SetIntValue('Width', int(width)), 'Width')
        self._ok(self.cam.MV_CC_SetIntValue('Height', int(height)), 'Height')

    def reset_roi(self):
        self.set_roi(0, 0, 1440, 1080)

    def set_pixel_format(self, name):
        if name not in PIXEL_FORMATS:
            raise CameraError('不支持的格式别名 %s，可选 %s' % (name, list(PIXEL_FORMATS)))
        self._ok(self.cam.MV_CC_SetEnumValue('PixelFormat', PIXEL_FORMATS[name]),
                 'PixelFormat=%s' % name)
        self.cur_format = PIXEL_FORMATS[name]

    def get(self, key):
        """读取一个 float 参数的当前值"""
        v = MVCC_FLOATVALUE()
        self._ok(self.cam.MV_CC_GetFloatValue(key, v), 'Get %s' % key)
        return v.fCurValue

    # ---------------- 取流 ----------------
    def start(self):
        self._ok(self.cam.MV_CC_StartGrabbing(), 'StartGrabbing')

    def stop(self):
        self.cam.MV_CC_StopGrabbing()

    def grab(self, timeout_ms=1000, convert=True):
        """取一帧，返回 BGR ndarray；超时返回 None"""
        st = MV_FRAME_OUT()
        ret = self.cam.MV_CC_GetImageBuffer(st, timeout_ms)
        if ret != 0:
            return None
        try:
            info = st.stFrameInfo
            w, h, pf = info.nWidth, info.nHeight, info.enPixelType
            size = info.nFrameLen
            buf = (c_ubyte * size).from_address(addressof(st.pBufAddr.contents))
            arr = np.frombuffer(buf, dtype=np.uint8, count=size)
            if convert:
                return self._decode(arr, w, h, pf)
            return arr.copy(), w, h, pf
        finally:
            self.cam.MV_CC_FreeImageBuffer(st)

    def _decode(self, arr, w, h, pf):
        if pf == PIXEL_FORMATS['BayerRG8']:
            # 实测该相机标称 BayerRG8，实际是 BGGR 排列
            return cv2.cvtColor(arr.reshape(h, w), cv2.COLOR_BayerBG2BGR)
        if pf == PIXEL_FORMATS['BGR8_Planar']:
            # 实测平面顺序为 R,G,B，需反序
            planes = arr.reshape(3, h, w)[::-1].transpose(1, 2, 0)
            return np.ascontiguousarray(planes)
        if pf == PIXEL_FORMATS['RGB8_Planar']:
            planes = arr.reshape(3, h, w).transpose(1, 2, 0)
            return cv2.cvtColor(np.ascontiguousarray(planes), cv2.COLOR_RGB2BGR)
        if pf == PIXEL_FORMATS['Mono8']:
            return cv2.cvtColor(arr.reshape(h, w), cv2.COLOR_GRAY2BGR)
        raise CameraError('未实现的像素格式 0x%08x' % pf)
