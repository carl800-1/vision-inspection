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
    # 主轴判据：环心裁剪小图中央区域与轴模板 NCC，高于阈值视为“有主轴”
    shaft_present: bool = False
    shaft_score: float = 0.0
    notes: str = ""
    overlay: object = field(default=None, repr=False)

    @property
    def state_index(self) -> int:
        """对应三张状态图：1=无环, 2=有环, 3=放A。"""
        if self.a_present:
            return 3
        return 2 if self.ring_state == "PLACED" else 1


# --------------------------------------------------------------------------- #
# 裁剪模板法（新方案）：自动定位环心 → 裁固定小图 → 模板NCC判定
# --------------------------------------------------------------------------- #
def locate_ring_center(img_bgr, model: Model, cfg: DetectConfig):
    """在训练标定的环心附近做局部微调搜索，返回 (cx, cy, score, ok)。

    相机固定后环的大致位置恒定，只在种子位置附近小范围搜索，避免全图
    NCC 把"有环小图"误匹配到别的金属纹理上（那会让环心跑偏、判错状态）。
    状态3环被工件A遮挡时匹配分数很低，ok=False，调用方回退到种子环心。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rc = getattr(model, "ring_center", [0.5, 0.5]) if model else [0.5, 0.5]
    ex, ey = int((rc[0] or 0.5) * w), int((rc[1] or 0.5) * h)
    if not model or not model.has_locator:
        return ex, ey, 0.0, False
    # 局部搜索窗口：围绕种子环心，半径取裁剪半径的1.5倍或短边12%
    win = int(max(cfg.crop_half * 1.5, min(h, w) * 0.12))
    x0, y0 = max(0, ex - win), max(0, ey - win)
    x1, y1 = min(w, ex + win), min(h, ey + win)
    search = gray[y0:y1, x0:x1]

    best_score, best_loc, best_shape = -1.0, None, None
    for t in model.template_images("locator"):
        tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
        # 定位模板来自工作尺寸，运行时图可能同尺寸，直接比
        score, loc = match_score(search, tg, cfg.locator_scale_search)
        if score > best_score:
            best_score, best_loc, best_shape = score, loc, tg.shape
    if best_loc is None:
        return ex, ey, 0.0, False
    th, tw = best_shape
    cx = int(x0 + best_loc[0] + tw / 2)
    cy = int(y0 + best_loc[1] + th / 2)
    return cx, cy, float(max(0.0, best_score)), best_score >= cfg.locator_th


def crop_at_center(img_bgr, cx, cy, half):
    """以 (cx,cy) 为中心裁 2*half 见方的小图，越界处补边。"""
    h, w = img_bgr.shape[:2]
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_REPLICATE)
        return padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return img_bgr[y0:y1, x0:x1]


def _ncc_scores(gray_crop, templates, scale_search=0.0):
    scores = []
    for t in templates:
        tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
        scores.append(match_score(gray_crop, _fit_template(tg, gray_crop),
                                  scale_search)[0])
    return scores


def center_square(gray, ratio: float = 0.35):
    """取灰度图中央正方形子区（主轴端面所在区域）。"""
    if gray is None or gray.ndim != 2:
        return None
    h, w = gray.shape[:2]
    side = max(16, int(min(h, w) * max(0.1, min(0.95, ratio))))
    x0, y0 = (w - side) // 2, (h - side) // 2
    return gray[y0:y0 + side, x0:x0 + side]


def shaft_ncc(gray_crop, model: Optional[Model], cfg: DetectConfig) -> float:
    """主轴端面 NCC 分数。

    优先用训练时保存的 shaft 模板；老模型没有 shaft 模板时，自动从 absent
    （无环）模板的中央子区生成，保证旧模型无需重训即可升级。
    """
    target = center_square(gray_crop, getattr(cfg, "shaft_crop_ratio", 0.35))
    if target is None:
        return 0.0
    templates = model.template_images("shaft") if model else []
    if not templates and model:
        for t in model.template_images("absent"):
            tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
            sub = center_square(tg, getattr(cfg, "shaft_crop_ratio", 0.35))
            if sub is not None:
                templates.append(sub)
    if not templates:
        return 0.0
    scores = []
    for t in templates:
        tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
        fitted = _fit_template(tg, target)
        scores.append(match_score(target, fitted, 0.0)[0])
    return max(scores) if scores else 0.0


# --------------------------------------------------------------------------- #
# 环检测器
# --------------------------------------------------------------------------- #
class RingDetector:
    def __init__(self, cfg: DetectConfig, model: Optional[Model] = None):
        self.cfg = cfg
        self.model = model

    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        # ===== 新方案：裁剪模板法 =====
        if self.cfg.ring_mode == "crop_template":
            return self._detect_crop_template(img_bgr)

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

    def _detect_crop_template(self, img_bgr: np.ndarray) -> DetectionResult:
        """裁剪模板法：定位环心→裁小图→present/absent模板NCC。"""
        h, w = img_bgr.shape[:2]
        res = DetectionResult()
        res.overlay = img_bgr.copy()
        if not self.model or not self.model.has_ring_template:
            res.notes = "裁剪模板法缺模型（请先训练）"
            return res

        half = int(getattr(self.model, "crop_half", 0) or self.cfg.crop_half)
        rc = getattr(self.model, "ring_center", [0.0, 0.0])
        def_x, def_y = (int(rc[0] * w), int(rc[1] * h)) if rc and rc[0] > 0 else (int(w / 2), int(h / 2))
        if getattr(self.cfg, "use_locator", False):
            cx, cy, loc_score, loc_ok = locate_ring_center(img_bgr, self.model, self.cfg)
            if not loc_ok:
                cx, cy = def_x, def_y
        else:
            # 相机固定：直接用训练标定的环心（最稳）
            cx, cy, loc_score, loc_ok = def_x, def_y, 0.0, True
        res.geometry = {"cx": cx, "cy": cy, "locator_score": loc_score,
                        "locator_ok": loc_ok, "crop_half": half}
        crop = crop_at_center(img_bgr, cx, cy, half)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        ps = _ncc_scores(gray, self.model.template_images("present"),
                         self.cfg.scale_search)
        ns = _ncc_scores(gray, self.model.template_images("absent"),
                         self.cfg.scale_search)
        p = max(ps) if ps else 0.0
        n = max(ns) if ns else 0.0
        res.ring_present_score, res.ring_absent_score = p, n

        # 主轴端面：中央子区 NCC（状态1≈1.0，状态2≈0.0，状态3<0）
        res.shaft_score = shaft_ncc(gray, self.model, self.cfg)
        res.shaft_present = res.shaft_score >= getattr(self.cfg, "shaft_ncc_th", 0.60)

        if p >= self.cfg.ring_ncc_th and p - n >= self.cfg.template_margin:
            res.ring_state = "PLACED"
        elif n >= self.cfg.ring_ncc_th and n - p >= self.cfg.template_margin:
            res.ring_state = "ABSENT"
        else:
            # 两个模板都不够像：状态3遮挡时会走到这里，用定位分数辅助判断
            res.ring_state = "UNCERTAIN"

        # 标注：裁剪框 + 环心 + 分数
        cv2.rectangle(res.overlay, (cx - half, cy - half),
                      (cx + half, cy + half), (0, 200, 255), 2)
        cv2.circle(res.overlay, (cx, cy), 4, (0, 255, 0), -1)
        res.notes = (f"定位={'OK' if loc_ok else '回退'}({loc_score:.2f}) "
                     f"环NCC 有={p:.2f} 无={n:.2f} 轴={res.shaft_score:.2f}")
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
        # 新方案：裁剪模板法下，A 判定也用同一环心裁图 + A模板NCC
        if self.cfg.ring_mode == "crop_template":
            return self._detect_crop_template(img_bgr, res)

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

    def _detect_crop_template(self, img_bgr, res: DetectionResult) -> DetectionResult:
        """A判定（裁剪模板法）：环心裁图与A模板比NCC。"""
        if not self.model or not self.model.has_a_template:
            res.notes = (res.notes + " / " if res.notes else "") + "缺A模板"
            return res
        geo = res.geometry
        cx, cy = geo.get("cx", int(img_bgr.shape[1] / 2)), geo.get("cy", int(img_bgr.shape[0] / 2))
        half = geo.get("crop_half", self.cfg.crop_half)
        crop = crop_at_center(img_bgr, cx, cy, half)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scores = _ncc_scores(gray, self.model.template_images("a"), self.cfg.scale_search)
        res.a_score = max(scores) if scores else 0.0
        res.a_present = res.a_score >= self.cfg.a_ncc_th
        res.notes = (res.notes + " / " if res.notes else "") + f"A_NCC={res.a_score:.2f}"
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
        text = (f"state {res.state_index} | ring={res.ring_state} "
                f"| shaft={'YES' if res.shaft_present else 'NO'} "
                f"| A={'YES' if res.a_present else 'NO'}")
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
