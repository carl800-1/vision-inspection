# -*- coding: utf-8 -*-
"""环形轴承判定核心：输入 BGR 图像，输出结构化判定结果。

判据来自实测：有环时环外缘存在深黑环形缝（暗谷亮度 11~32），
无环时只有平缓沟槽（暗谷 77~84）；阈值默认 55。
"""
import cv2
import numpy as np

DEFAULT_CONFIG = {
    "roi_ratio": 0.60,      # 中心 ROI 占比
    "valley_th": 55.0,      # 暗谷判定阈值
    "param2": 45,           # 霍夫累加器阈值
    "radius_lo": 0.18,      # 霍夫半径下限（短边比例）
    "radius_hi": 0.42,      # 霍夫半径上限（短边比例）
    "band_lo": 0.30,        # 剖面搜索带下限（r0 倍数）
    "band_hi": 1.80,        # 剖面搜索带上限（r0 倍数）
    "margin_min": 0.20,     # 置信裕度下限，低于此值判"待复检"
}


def detect_ring(img_bgr, config=None):
    """判定单帧图像中心是否放置环形轴承。

    返回 dict:
        verdict      'PLACED' | 'ABSENT' | 'UNCERTAIN'
        has_ring     bool / None
        circle       (cx, cy, r) 或 None
        valley       暗谷平均亮度
        valley_r     暗谷半径(px)
        peak         亮峰平均亮度
        peak_r       亮峰半径(px)
        contrast     落差
        margin       相对阈值裕度 abs(valley-th)/th
        overlay      带标注的 BGR 副本
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    result = {
        "verdict": "UNCERTAIN", "has_ring": None, "circle": None,
        "valley": None, "valley_r": None, "peak": None, "peak_r": None,
        "contrast": None, "margin": 0.0, "overlay": img_bgr.copy(),
    }
    if img_bgr is None:
        return result

    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < 40:
        return result

    # 1) 中心 ROI + 灰度 + 中值模糊
    x0 = int(w * (1 - cfg["roi_ratio"]) / 2)
    y0 = int(h * (1 - cfg["roi_ratio"]) / 2)
    x1 = int(w * (1 + cfg["roi_ratio"]) / 2)
    y1 = int(h * (1 + cfg["roi_ratio"]) / 2)
    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # 2) 霍夫圆定位中心圆
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, int(min_dim * 0.16)),
        param1=120, param2=cfg["param2"],
        minRadius=int(min_dim * cfg["radius_lo"]),
        maxRadius=int(min_dim * cfg["radius_hi"]),
    )

    found = []
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            gx, gy = cx + x0, cy + y0
            found.append((gx, gy, r))

    # 取距图像中心最近的圆
    cx_img, cy_img = w / 2, h / 2
    found.sort(key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
    if found:
        gx, gy, r0 = found[0]
    else:
        gx, gy, r0 = int(cx_img), int(cy_img), 0
    result["circle"] = (gx, gy, r0)

    # 3) 径向亮度剖面
    gb = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 7)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2).astype(int)
    if r0 > 0:
        band = range(max(2, int(r0 * cfg["band_lo"])),
                     max(4, int(r0 * cfg["band_hi"])), 2)
    else:
        band = range(int(min_dim * 0.20), max(int(min_dim * 0.50), 6), 2)
    band = list(band)
    if len(band) < 3:
        return result

    prof = np.array([gb[rr == r].mean() for r in band])
    sm = np.convolve(prof, np.ones(5) / 5, "same")
    valley_i = int(np.argmin(sm))
    valley = float(sm[valley_i])
    peak = float(sm[valley_i:].max())
    peak_i = valley_i + int(np.argmax(sm[valley_i:]))

    th = cfg["valley_th"]
    margin = abs(valley - th) / th
    result.update({
        "valley": valley, "valley_r": band[valley_i],
        "peak": peak, "peak_r": band[peak_i],
        "contrast": peak - valley, "margin": margin,
    })
    if margin < cfg["margin_min"]:
        result["verdict"] = "UNCERTAIN"
    elif valley < th:
        result["verdict"] = "PLACED"
        result["has_ring"] = True
    else:
        result["verdict"] = "ABSENT"
        result["has_ring"] = False

    # 4) 标注图
    ov = img_bgr.copy()
    for (fx, fy, fr) in found:
        cv2.circle(ov, (fx, fy), fr, (0, 0, 255), 2)
    cv2.circle(ov, (gx, gy), band[valley_i], (255, 0, 0), 2)
    cv2.circle(ov, (gx, gy), band[peak_i], (0, 255, 255), 2)
    cv2.circle(ov, (gx, gy), 3, (0, 255, 0), -1)
    colors = {"PLACED": (0, 0, 255), "ABSENT": (160, 160, 160), "UNCERTAIN": (0, 165, 255)}
    texts = {"PLACED": "PLACED", "ABSENT": "ABSENT", "UNCERTAIN": "RECHECK"}
    cv2.putText(ov, texts[result["verdict"]], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, colors[result["verdict"]], 2, cv2.LINE_AA)
    result["overlay"] = ov
    return result
