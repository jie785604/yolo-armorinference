#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板四角点 → PnP 位姿解算

坐标系约定（装甲板系）：
    原点：板面中心
    X：沿板的长边向右（从相机看向装甲板时）
    Y：沿板的短边向上
    Z：垂直板面向外，指向相机

图像角点顺序实测为「左上、左下、右下、右上」（模型 keypoints 的原始顺序，
沿板边顺时针绕行），三维点按同样顺序给出。
"""
import cv2
import numpy as np

# 装甲板实际尺寸（毫米），量的是两侧 LED 灯条外沿
DEFAULT_ARMOR_W_MM = 230.0
DEFAULT_ARMOR_H_MM = 55.0


def armor_object_points(width_mm=DEFAULT_ARMOR_W_MM, height_mm=DEFAULT_ARMOR_H_MM):
    """返回 4 个三维角点：左上、左下、右下、右上（与模型关键点顺序一致）"""
    hw, hh = width_mm / 2.0, height_mm / 2.0
    return np.array([
        [-hw, hh, 0.0],
        [-hw, -hh, 0.0],
        [hw, -hh, 0.0],
        [hw, hh, 0.0],
    ], dtype=np.float32)


def order_corners_geometric(pts):
    """把任意顺序的 4 个点排成 左上、左下、右下、右上

    做法：按 y 坐标分上下两行，每行再按 x 排序。
    适合竖直放置、没有大幅倾斜的装甲板；若装甲板侧倾超过约 45 度会排错，
    此时应在运行时用模型关键点的原始顺序并在 --debug-order 下确认。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    order = np.argsort(pts[:, 1])
    top = pts[order[:2]]
    bottom = pts[order[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, bl, br, tr], dtype=np.float32)


def solve_pose(corners_img, camera_matrix, dist_coeffs,
               width_mm=DEFAULT_ARMOR_W_MM, height_mm=DEFAULT_ARMOR_H_MM):
    """解算位姿，返回 (rvec, tvec, reproj_error_px)

    用 SOLVEPNP_IPPE：装甲板是平面目标，IPPE 会给出两个候选解（一个镜像），
    这里取重投影误差更小的那个。
    """
    obj = armor_object_points(width_mm, height_mm)
    img = np.asarray(corners_img, dtype=np.float32).reshape(4, 2)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None, None, None
    best = int(np.argmin([e[0] if hasattr(e, '__len__') else e for e in errs]))
    rvec = rvecs[best].reshape(3, 1)
    tvec = tvecs[best].reshape(3, 1)
    err = float(errs[best][0] if hasattr(errs[best], '__len__') else errs[best])
    return rvec, tvec, err


def reprojection_error(rvec, tvec, corners_img, camera_matrix, dist_coeffs,
                       width_mm=DEFAULT_ARMOR_W_MM, height_mm=DEFAULT_ARMOR_H_MM):
    obj = armor_object_points(width_mm, height_mm)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    img = np.asarray(corners_img, dtype=np.float32).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img, axis=1)))


def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, length_mm=50.0):
    """在图像上画装甲板坐标系的三根轴（X 红、Y 绿、Z 蓝）"""
    pts3d = np.array([
        [0, 0, 0],
        [length_mm, 0, 0],
        [0, length_mm, 0],
        [0, 0, length_mm],
    ], dtype=np.float32)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2).astype(int)
    o = tuple(proj[0])
    cv2.line(frame, o, tuple(proj[1]), (0, 0, 255), 2)
    cv2.line(frame, o, tuple(proj[2]), (0, 255, 0), 2)
    cv2.line(frame, o, tuple(proj[3]), (255, 0, 0), 2)
