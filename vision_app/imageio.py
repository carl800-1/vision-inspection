# -*- coding: utf-8 -*-
"""图像读写与 ROI 工具：兼容中文路径。"""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """cv2.imread 在中文路径下会失败，这里用 fromfile + imdecode。"""
    if not path or not os.path.isfile(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def list_images(folder: str, recursive: bool = True) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(IMAGE_EXTS):
                    out.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            if name.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(folder, name))
    return sorted(out)


def norm_to_pixels(roi: Iterable[float], shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """归一化 ROI [x,y,w,h] → 像素 (x0,y0,x1,y1)，保证至少 1px。"""
    h, w = shape[:2]
    x, y, rw, rh = [float(v) for v in roi]
    x0 = int(round(x * w))
    y0 = int(round(y * h))
    x1 = int(round((x + rw) * w))
    y1 = int(round((y + rh) * h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    return x0, y0, x1, y1


def crop_norm(img: np.ndarray, roi: Iterable[float]):
    x0, y0, x1, y1 = norm_to_pixels(roi, img.shape)
    return img[y0:y1, x0:x1]


def match_score(gray: np.ndarray, template_gray: np.ndarray,
                scale_search: float = 0.0) -> Tuple[float, Tuple[int, int]]:
    """在 gray 中查找 template，返回最佳 NCC 分数与位置。"""
    if gray is None or template_gray is None or gray.ndim != 2 or template_gray.ndim != 2:
        return 0.0, (0, 0)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return 0.0, (0, 0)
    th, tw = template_gray.shape
    if th < 8 or tw < 8 or th > gray.shape[0] or tw > gray.shape[1]:
        return 0.0, (0, 0)

    best, best_loc = -1.0, (0, 0)
    scales = [1.0]
    if scale_search and scale_search > 0:
        scales = [1.0 - scale_search, 1.0, 1.0 + scale_search]
    for s in scales:
        sh, sw = max(8, int(th * s)), max(8, int(tw * s))
        if sh > gray.shape[0] or sw > gray.shape[1]:
            continue
        tpl = template_gray if s == 1.0 else cv2.resize(template_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _mn, mx, _mnloc, loc = cv2.minMaxLoc(res)
        if mx > best:
            best, best_loc = mx, loc
    return float(max(0.0, best)), best_loc


def largest_diff_roi(before: np.ndarray, after: np.ndarray,
                     pad_ratio: float = 0.12, diff_th: int = 22):
    """比较状态2(有环)与状态3(放A)两图，自动定位新出现的工件A区域。

    返回归一化 ROI [x,y,w,h]；无有效差异时返回 None。
    """
    if before is None or after is None or before.shape[:2] != after.shape[:2]:
        return None
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
    diff = cv2.absdiff(g1, g2)
    _ret, mask = cv2.threshold(diff, diff_th, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    hh, ww = g1.shape[:2]
    if w < ww * 0.02 or h < hh * 0.02:
        return None
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(ww, x + w + px), min(hh, y + h + py)
    roi = [round(x0 / ww, 4), round(y0 / hh, 4),
           round((x1 - x0) / ww, 4), round((y1 - y0) / hh, 4)]
    # 视野/构图不同会让全图都变成差异区，这种 ROI 没有定位价值
    if roi[2] * roi[3] > 0.72:
        return None
    return roi


def draw_label(img: np.ndarray, text: str, color=(0, 0, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, color, 2, cv2.LINE_AA)
    return out
