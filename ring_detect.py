# -*- coding: utf-8 -*-
"""用霍夫圆检测判断零件中心是否存在圆环。
输入: with ring.jpg / without ring.jpg (3072x4096)
输出: result_with.jpg / result_without.jpg + 检测结果打印
"""
import cv2
import numpy as np
import os

DESKTOP = r"C:\Users\Chenzhichao\Desktop"
OUT = r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42"

# ROI: 取图像中间 60%（减少四周螺栓孔、脚踏等干扰）
ROI_RATIO = 0.60
# 霍夫圆参数（按 3072x4096 原图调整）
MIN_RADIUS = 120
MAX_RADIUS = 320
PARAM1 = 120   # Canny 高阈值
PARAM2 = 55    # 累加器阈值（圆的"置信度"，越大越严格）
MIN_DIST = 150 # 圆心最小间距
DP = 1.2

def detect(path, out_path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    # 1. 裁剪中心 ROI
    x0, y0 = int(w * (1 - ROI_RATIO) / 2), int(h * (1 - ROI_RATIO) / 2)
    x1, y1 = int(w * (1 + ROI_RATIO) / 2), int(h * (1 + ROI_RATIO) / 2)
    roi = img[y0:y1, x0:x1]

    # 2. 灰度 + 中值模糊
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 9)

    # 3. 霍夫圆检测（只测 ROI）
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=PARAM1, param2=PARAM2,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    result = img.copy()
    found = []
    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            cx, cy, r = c
            # ROI 坐标映射回原图
            gx, gy = cx + x0, cy + y0
            found.append((gx, gy, r))
            cv2.circle(result, (gx, gy), r, (0, 0, 255), 6)
            cv2.circle(result, (gx, gy), 4, (0, 255, 0), -1)

    cv2.imwrite(out_path, result)

    name = os.path.basename(path)
    print(f"--- {name} ---")
    print(f"图像尺寸: {w}x{h}, ROI: [{x0}:{x1}, {y0}:{y1}]")
    print(f"检测到圆数量: {len(found)}")
    for i, (gx, gy, r) in enumerate(found):
        # 与图像中心的距离
        d = ((gx - w / 2) ** 2 + (gy - h / 2) ** 2) ** 0.5
        print(f"  圆{i+1}: 圆心=({gx},{gy}) 半径={r}px 距图像中心={d:.0f}px")
    # 判定：最靠近图像中心的圆且距离 < 150px 视为"中心圆环"
    center_circles = [c for c in found
                      if ((c[0] - w / 2) ** 2 + (c[1] - h / 2) ** 2) ** 0.5 < 150]
    verdict = "存在中心圆环" if center_circles else "未检测到中心圆环"
    print(f"判定: {verdict}")
    return found

for src, dst in [("with ring.jpg", "result_with.jpg"),
                 ("without ring.jpg", "result_without.jpg")]:
    detect(os.path.join(DESKTOP, src), os.path.join(OUT, dst))
