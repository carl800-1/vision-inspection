# -*- coding: utf-8 -*-
"""判断零件中心是否装有凸起金属圆环（Hikvision 相机拍摄）。

方法（不局限于霍夫圆）：
  1. 中心 60% ROI 裁剪 → 灰度 → 中值模糊；
  2. cv2.HoughCircles 定位中心圆（半径范围按图像尺寸等比例设置）；
  3. 判别用径向亮度剖面：以检出圆心为圆心，按半径逐圈求平均亮度。
     装了凸环的零件，环外缘有一条很暗的环形缝隙（阴影），剖面出现"深黑谷"；
     没装环只有平缓沟槽，最暗处也明显偏亮。
  4. 搜索区间绑定检出半径 r0（0.3~1.8 倍），跨拍摄距离可迁移。

判据：暗谷亮度 < VALLEY_TH (55)。实测：有环 11~32，无环 84。
"""
import cv2
import numpy as np
import os
import sys

ROI_RATIO = 0.60
PARAM1 = 120
PARAM2 = 45
MIN_DIST = 60
DP = 1.2
VALLEY_TH = 55
RING_MARGIN = 0.20   # 检出圆心偏离图像中心 < 短边 20% 才算中心圆


def detect(path, out_path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    min_dim = min(h, w)

    # ---- 1. 中心 ROI ----
    x0, y0 = int(w * (1 - ROI_RATIO) / 2), int(h * (1 - ROI_RATIO) / 2)
    x1, y1 = int(w * (1 + ROI_RATIO) / 2), int(h * (1 + ROI_RATIO) / 2)
    roi = img[y0:y1, x0:x1]

    # ---- 2. 霍夫圆定位中心圆 ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    r_min, r_max = int(min_dim * 0.18), int(min_dim * 0.42)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=PARAM1, param2=PARAM2, minRadius=r_min, maxRadius=r_max)

    result = img.copy()
    found = []
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            gx, gy = cx + x0, cy + y0
            found.append((gx, gy, r))
            cv2.circle(result, (gx, gy), r, (0, 0, 255), 2)
            cv2.circle(result, (gx, gy), 3, (0, 255, 0), -1)

    print(f"--- {os.path.basename(path)} ({w}x{h}) ---")
    print(f"半径搜索范围: {r_min}~{r_max}px, 检测到圆: {len(found)} 个")
    for i, (gx, gy, r) in enumerate(found):
        d = ((gx - w / 2) ** 2 + (gy - h / 2) ** 2) ** 0.5
        print(f"  圆{i+1}: 圆心=({gx},{gy}) 半径={r}px 距图像中心={d:.0f}px")

    # 取最靠近图像中心的圆作为中心圆
    center_hits = [c for c in found
                   if ((c[0] - w / 2) ** 2 + (c[1] - h / 2) ** 2) ** 0.5 < min_dim * RING_MARGIN]
    if center_hits:
        gx, gy, r0 = center_hits[0]
    else:
        gx, gy, r0 = w // 2, h // 2, 0

    # ---- 3. 径向亮度剖面（以中心圆心为圆心） ----
    gb = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2).astype(int)
    # 搜索带：检出半径的 0.3~1.8 倍；无检出圆时退回短边 0.4~1.0 半径
    if r0 > 0:
        band = range(int(r0 * 0.3), int(r0 * 1.8), 2)
    else:
        band = range(int(min_dim * 0.20), int(min_dim * 0.50), 2)
    prof = np.array([gb[rr == r].mean() for r in band])
    sm = np.convolve(prof, np.ones(5) / 5, "same")

    valley_i = int(np.argmin(sm))
    valley = sm[valley_i]
    valley_r = list(band)[valley_i]
    peak = sm[valley_i:].max()
    peak_r = list(band)[valley_i + int(np.argmax(sm[valley_i:]))]
    contrast = peak - valley

    has_ring = valley < VALLEY_TH
    verdict = "RING DETECTED" if has_ring else "NO RING"
    print(f"  中心圆: r0={r0}px, 搜索带 {list(band)[0]}~{list(band)[-1]}px")
    print(f"  剖面: 暗谷={valley:.0f}@r={valley_r}px (阈值 {VALLEY_TH}), "
          f"亮峰={peak:.0f}@r={peak_r}px, 落差={contrast:.0f}")
    print(f"  判定: {verdict}")

    # ---- 4. 标注 ----
    cv2.circle(result, (gx, gy), valley_r, (255, 0, 0), 2)   # 蓝: 暗谷(环缝)
    cv2.circle(result, (gx, gy), peak_r, (0, 255, 255), 2)   # 黄: 亮峰
    color = (0, 0, 255) if has_ring else (160, 160, 160)
    cv2.putText(result, verdict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, result)
    # 灰度标注版（肉眼比对更直观；判别计算本身就在灰度图上进行）
    g_name, g_ext = os.path.splitext(out_path)
    cv2.imwrite(g_name + "_gray" + g_ext, cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    return verdict


if __name__ == "__main__":
    jobs = [
        (r"C:\Users\Chenzhichao\.workbuddy\clipboard-images\clipboard-2026-11-24T03-37-39-131Z-147aa35c.png",
         r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42\result_with2.jpg"),
        (r"C:\Users\Chenzhichao\.workbuddy\clipboard-images\clipboard-2026-11-24T03-37-39-154Z-7bbf7569.png",
         r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42\result_without2.jpg"),
        (r"C:\Users\Chenzhichao\Desktop\with ring.jpg",
         r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42\result_with.jpg"),
        (r"C:\Users\Chenzhichao\Desktop\without ring.jpg",
         r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42\result_without.jpg"),
    ]
    if len(sys.argv) > 1:   # 可传任意图片路径单独检测
        jobs = [(sys.argv[1], os.path.join(
            r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42",
            "result_" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + ".jpg"))]
    for src, dst in jobs:
        detect(src, dst)
