# -*- coding: utf-8 -*-
"""检测器：几何法 + 示教模板，输出1f统一结果结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from .config import DetectConfig
from .imageio import crop_norm, match_score, draw_label
from .model import Model


# --------------------------------------------------------------------------- #
# 几何法：沿用已验证的径向亮度剖面判据
# --------------------------------------------------------------------------- #
def _fit_template(template, gray):
    """模板来自原图、检测在缩放图上；先等比缩到搜索区尺寸。"""
    if template.shape[:2] == gray.shape[:2]:
        return template
    return cv2.resize(template, (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_AREA)


def geometry_detect(img_bgr: np.ndarray, cfg: DetectConfig) -> Dict:
    """返回 dict(verdict, valley, peak, contrast, margin, circle, overlay)。"""
    out = {
        "verdict": "UNCERTAIN", "valley": None, "peak": None,
        "contrast": None, "margin": 0.0, "circle": None, "overlay": img_bgr.copy(),
    }
    if img_bgr is None:
        return out
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < 40:
        return out

    x0 = int(w * (1 - cfg.roi_ratio) / 2)
    y0 = int(h * (1 - cfg.roi_ratio) / 2)
    x1 = int(w * (1 + cfg.roi_ratio) / 2)
    y1 = int(h * (1 + cfg.roi_ratio) / 2)
    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, int(min_dim * 0.16)),
        param1=120, param2=cfg.hough_param2,
        minRadius=int(min_dim * cfg.radius_lo),
        maxRadius=int(min_dim * cfg.radius_hi),
    )
    found = []
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            found.append((cx + x0, cy + y0, r))
    found.sort(key=lambda c: (c[0] - w / 2) ** 2 + (c[1] - h / 2) ** 2)
    if found:
        gx, gy, r0 = found[0]
    else:
        gx, gy, r0 = int(w / 2), int(h / 2), 0
    out["circle"] = (int(gx), int(gy), int(r0))

    full_gray = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 7)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2).astype(int)
    if r0 > 0:
        band = list(range(max(2, int(r0 * cfg.band_lo)),
                          max(4, int(r0 * cfg.band_hi)), 2))
    else:
        band = list(range(int(min_dim * 0.20),
                          max(int(min_dim * 0.50), 6), 2))
    if len(band) < 3:
        return out

    prof = np.array([full_gray[rr == r].mean() for r in band])
    sm = np.convolve(prof, np.ones(5) / 5, "same")
    # 找“暗谷→随后亮峰”落差最大的点，而不是全带最暗点
    # （避免视野边缘变暗时把最外圈误当暗谷）
    best_score, vi, pi = -1e9, 0, 0
    win = max(3, min(28, len(band) // 3))
    for i in range(len(band) - win):
        j = i + 1 + int(np.argmax(sm[i + 1:i + 1 + win]))
        score = sm[j] - sm[i]
        if score > best_score:
            best_score, vi, pi = score, i, j
    valley = float(sm[vi])
    peak = float(sm[pi])
    margin = abs(valley - cfg.valley_th) / cfg.valley_th

    out.update({
        "valley": valley, "valley_r": band[vi],
        "peak": peak, "peak_r": band[pi],
        "contrast": peak - valley, "margin": margin,
    })
    if margin < cfg.margin_min:
        out["verdict"] = "UNCERTAIN"
    elif valley < cfg.valley_th:
        out["verdict"] = "PLACED"
    else:
        out["verdict"] = "ABSENT"

    ov = img_bgr.copy()
    for fx, fy, fr in found:
        cv2.circle(ov, (int(fx), int(fy)), int(fr), (0, 0, 255), 2)
    cv2.circle(ov, (gx, gy), band[vi], (255, 0, 0), 2)
    cv2.circle(ov, (gx, gy), band[pi], (0, 255, 255), 2)
    cv2.circle(ov, (gx, gy), 3, (0, 255, 0), -1)
    out["overlay"] = ov
    return out


# --------------------------------------------------------------------------- #
# 统一结果
# --------------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    ring_state: str = "UNCERTAIN"       # PLACED / ABSENT / UNCERTAIN
    a_present: bool = False
    geometry: Dict = field(default_factory=dict)
    ring_present_score: float = 0.0
    ring_absent_score: float = 0.0
    a_score: float = 0.0
    a_hole_ok: bool = False
    a_edge_ok: bool = False
    a_hole_center = None
    center_edge: float = 0.0
    center_clear: bool = False
    notes: str = ""
    overlay: object = field(default=None, repr=False)

    @property
    def state_index(self) -> int:
        """对应三张状态图：1=无环, 2=有环, 3=放A。"""
        if self.a_present:
            return 3
        return 2 if self.ring_state == "PLACED" else 1


# --------------------------------------------------------------------------- #
# 环检测器
# --------------------------------------------------------------------------- #
class RingDetector:
    def __init__(self, cfg: DetectConfig, model: Optional[Model] = None):
        self.cfg = cfg
        self.model = model

    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        geo = geometry_detect(img_bgr, self.cfg)
        res = DetectionResult(geometry=geo, overlay=geo["overlay"].copy())

        if not self.model or not self.model.has_ring_template:
            res.ring_state = geo["verdict"]
            res.notes = "几何法（无示教模板）"
            return res

        roi = crop_norm(img_bgr, self.model.ring_roi)
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        def best(key):
            scores = [match_score(gray, _fit_template(
                cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t, gray),
                self.cfg.scale_search)[0]
                      for t in self.model.template_images(key)]
            return max(scores) if scores else 0.0

        ps, ns = best("present"), best("absent")
        res.ring_present_score, res.ring_absent_score = ps, ns

        template_state = "UNCERTAIN"
        if ps >= self.cfg.template_th and ps - ns >= self.cfg.template_margin:
            template_state = "PLACED"
        elif ns >= self.cfg.template_th and ns - ps >= self.cfg.template_margin:
            template_state = "ABSENT"

        mode = self.cfg.ring_mode
        if mode == "geometry":
            res.ring_state = geo["verdict"]
        elif mode == "template":
            res.ring_state = template_state
        else:  # hybrid：两者一致直接采纳；冲突时以模板为准，几何法作为旁证
            if template_state == geo["verdict"]:
                res.ring_state = template_state
            elif template_state != "UNCERTAIN":
                res.ring_state = template_state
                res.notes = f"模板优先（几何={geo['verdict']}）"
            else:
                res.ring_state = geo["verdict"]
                res.notes = "模板不显著，回退几何法"
        return res


# --------------------------------------------------------------------------- #
# 工件 A 检测器
# --------------------------------------------------------------------------- #
class ADetector:
    """工件A判定：偏心孔 / 横边 / 模板投票 + 中心轴与环遮挡确认。"""

    def __init__(self, cfg: DetectConfig, model: Optional[Model] = None):
        self.cfg = cfg
        self.model = model

    def _draw_roi(self, img, roi, color):
        h, w = img.shape[:2]
        x0, y0, x1, y1 = [
            int(roi[0] * w), int(roi[1] * h),
            int((roi[0] + roi[2]) * w), int((roi[1] + roi[3]) * h),
        ]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

    def _center_clear(self, img_bgr, res):
        roi = crop_norm(img_bgr, self.cfg.center_roi)
        if roi.size == 0:
            return False
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        g = cv2.GaussianBlur(g, (3, 3), 0)
        edges = cv2.Canny(g, 60, 160)
        density = float((edges > 0).mean())
        res.center_edge = density
        res.center_clear = density < self.cfg.center_edge_th
        self._draw_roi(res.overlay, self.cfg.center_roi,
                       (0, 255, 255) if res.center_clear else (128, 128, 128))

    def _hole_feature(self, img_bgr, res):
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        roi = crop_norm(img_bgr, self.cfg.a_hole_roi)
        if roi.size == 0:
            return False
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        g = cv2.medianBlur(g, 5)
        circles = cv2.HoughCircles(
            g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(12, int(short * 0.06)),
            param1=100, param2=self.cfg.a_hole_param2,
            minRadius=max(4, int(short * self.cfg.a_hole_r_lo)),
            maxRadius=max(6, int(short * self.cfg.a_hole_r_hi)))
        if circles is None:
            return False
        ox, oy, ow, oh = self.cfg.a_hole_roi
        for cx, cy, r in np.round(circles[0]).astype(int):
            fx, fy = cx + ox * w, cy + oy * h
            offset = ((fx - w / 2) ** 2 + (fy - h / 2) ** 2) ** 0.5 / short
            if offset < self.cfg.a_hole_min_offset:
                continue
            res.a_hole_ok = True
            res.a_hole_center = (int(fx), int(fy), int(r))
            cv2.circle(res.overlay, (int(fx), int(fy)), int(r), (255, 0, 255), 2)
            self._draw_roi(res.overlay, self.cfg.a_hole_roi, (255, 0, 255))
            return True
        return False

    def _edge_feature(self, img_bgr, res):
        roi = crop_norm(img_bgr, self.cfg.a_edge_roi)
        if roi.size == 0:
            return False
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        g = cv2.GaussianBlur(g, (3, 3), 0)
        edges = cv2.Canny(g, self.cfg.a_edge_th, self.cfg.a_edge_th * 2)
        min_len = self.cfg.a_edge_min_len or int(min(g.shape) * 0.12)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12,
                                minLineLength=min_len, maxLineGap=8)
        if lines is None:
            return False
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(y2 - y1) <= max(2, abs(x2 - x1) * 0.18):
                res.a_edge_ok = True
                ox, oy, ow, oh = self.cfg.a_edge_roi
                h, w = img_bgr.shape[:2]
                cv2.line(res.overlay,
                         (int(x1 + ox * w), int(y1 + oy * h)),
                         (int(x2 + ox * w), int(y2 + oy * h)),
                         (0, 255, 0), 2)
                self._draw_roi(res.overlay, self.cfg.a_edge_roi, (0, 255, 0))
                return True
        return False

    def _template_feature(self, img_bgr, res):
        if not self.model or not self.model.has_a_template:
            return False
        roi = crop_norm(img_bgr, self.model.a_roi)
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        scores = []
        for t in self.model.template_images("a"):
            tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
            scores.append(match_score(gray, _fit_template(tg, gray),
                                      self.cfg.scale_search)[0])
        res.a_score = max(scores) if scores else 0.0
        self._draw_roi(res.overlay, self.model.a_roi,
                       (0, 165, 255) if res.a_score >= self.cfg.a_score_th
                       else (128, 128, 128))
        return res.a_score >= self.cfg.a_score_th

    def detect(self, img_bgr: np.ndarray, res: DetectionResult) -> DetectionResult:
        self._center_clear(img_bgr, res)
        votes = []
        mode = self.cfg.a_feature_mode
        if mode in ("hole", "hybrid"):
            votes.append(self._hole_feature(img_bgr, res))
        if mode in ("edge", "hybrid"):
            votes.append(self._edge_feature(img_bgr, res))
        if mode in ("template", "hybrid"):
            votes.append(self._template_feature(img_bgr, res))

        if mode == "hybrid":
            res.a_present = sum(votes) >= min(self.cfg.a_min_votes, len(votes))
        elif votes:
            res.a_present = votes[0]

        if self.cfg.require_center_clear:
            res.a_present = res.a_present and res.center_clear

        res.notes = (res.notes + " / " if res.notes else "") + (
            f"A孔={'Y' if res.a_hole_ok else 'N'} "
            f"A横边={'Y' if res.a_edge_ok else 'N'} "
            f"中心清空={'Y' if res.center_clear else 'N'}({res.center_edge:.3f})")
        return res


# --------------------------------------------------------------------------- #
# 汇总管线
# --------------------------------------------------------------------------- #
class VisionPipeline:
    """检测 + 标注，一帧调用一次。"""
    def __init__(self, cfg: DetectConfig, model: Optional[Model] = None):
        self.ring = RingDetector(cfg, model)
        self.a = ADetector(cfg, model)
        self.cfg = cfg
        self.model = model

    def set_model(self, model: Optional[Model]) -> None:
        self.model = model
        self.ring.model = model
        self.a.model = model

    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        h, w = img_bgr.shape[:2]
        scale = 1.0
        max_side = int(getattr(self.cfg, "max_side", 0) or 0)
        if max_side > 0 and max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            work = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        else:
            work = img_bgr

        res = self.ring.detect(work)
        res = self.a.detect(work, res)
        color = {"PLACED": (0, 170, 0), "ABSENT": (0, 0, 200),
                 "UNCERTAIN": (0, 165, 255)}[res.ring_state]
        text = f"state {res.state_index} | ring={res.ring_state} | A={'YES' if res.a_present else 'NO'}"
        res.overlay = draw_label(res.overlay, text, color)

        if scale < 1.0:
            res.overlay = cv2.resize(res.overlay, (w, h), interpolation=cv2.INTER_LINEAR)
            if res.geometry.get("circle"):
                cx, cy, r = res.geometry["circle"]
                res.geometry["circle"] = (int(cx / scale), int(cy / scale), int(r / scale))
            for key in ("valley_r", "peak_r"):
                if res.geometry.get(key) is not None:
                    res.geometry[key] = int(res.geometry[key] / scale)
        return res
