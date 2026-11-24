# -*- coding: utf-8 -*-
"""自学习：导入照片 → 自动定位ROI → 生成模板 → 保存模型。

照片标签（三态）：
    state1 / absent  ：无环，未放工件A
    state2 / present ：已放环，未放工件A
    state3 / a       ：已放环并放置工件A（环被遮挡）

可用 Trainer.add_image 逐张导入，也可用 DatasetManager 从目录批量导入。
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectConfig
from .detectors import geometry_detect
from .imageio import imread_unicode, largest_diff_roi, norm_to_pixels
from .model import Model, save_template_image

LABEL_ALIAS = {
    "state1": "absent", "1": "absent", "absent": "absent", "no_ring": "absent",
    "state2": "present", "2": "present", "present": "present", "with_ring": "present",
    "state3": "a", "3": "a", "a": "a", "workpiece_a": "a",
}


@dataclass
class Trainer:
    cfg: DetectConfig
    images: Dict[str, List[Tuple[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    report: List[str] = field(default_factory=list)

    def add_image(self, path: str, label: str) -> bool:
        """导入一张照片。label 见模块顶部说明。返回是否成功。"""
        key = LABEL_ALIAS.get(str(label).strip().lower())
        if key is None:
            self.report.append(f"跳过未知标签：{label} ({path})")
            return False
        img = imread_unicode(path)
        if img is None:
            self.report.append(f"图片读取失败：{path}")
            return False
        self.images[key].append((path, img))
        return True

    def add_folder(self, folder: str, label: str) -> int:
        from .imageio import list_images
        n = 0
        for p in list_images(folder, recursive=True):
            n += int(self.add_image(p, label))
        return n

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.images.items()}

    def _first_matching_pair(self, key_a: str, key_b: str):
        for pa, ia in self.images.get(key_a, []):
            for pb, ib in self.images.get(key_b, []):
                if ia.shape[:2] == ib.shape[:2]:
                    return ia, ib
        return None

    def _template_set(self, key: str, roi: List[float], model_path: str,
                      max_n: int = 6) -> List[str]:
        crops = []
        for _path, img in self.images.get(key, [])[:max_n * 3]:
            x0, y0, x1, y1 = norm_to_pixels(roi, img.shape)
            crop = img[y0:y1, x0:x1]
            if crop.shape[0] >= 12 and crop.shape[1] >= 12:
                crops.append(crop)
            if len(crops) >= max_n:
                break
        if not crops:
            return []
        # 统一尺寸：取中位数尺寸，避免不同分辨率照片训练出的模板尺度不一致
        hs = sorted(c.shape[0] for c in crops)
        ws = sorted(c.shape[1] for c in crops)
        th, tw = hs[len(hs) // 2], ws[len(ws) // 2]
        out = []
        for i, c in enumerate(crops):
            if c.shape[:2] != (th, tw):
                c = cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)
            out.append(save_template_image(Model(), model_path, key, c, i))
        return out

    def _derive_threshold(self) -> Optional[float]:
        valleys = {}
        for key in ("absent", "present"):
            vals = []
            for _p, img in self.images.get(key, []):
                work = img
                max_side = int(getattr(self.cfg, "max_side", 0) or 0)
                if max_side > 0 and max(img.shape[:2]) > max_side:
                    scale = max_side / float(max(img.shape[:2]))
                    work = cv2.resize(img, (int(img.shape[1] * scale),
                                            int(img.shape[0] * scale)),
                                      interpolation=cv2.INTER_AREA)
                r = geometry_detect(work, self.cfg)
                if r["valley"] is not None:
                    vals.append(r["valley"])
            if vals:
                valleys[key] = float(np.median(vals))
        if len(valleys) == 2 and valleys["present"] < valleys["absent"]:
            th = (valleys["present"] + valleys["absent"]) / 2.0
            self.report.append(
                f"自动阈值：有环={valleys['present']:.1f} 无环={valleys['absent']:.1f} → {th:.1f}")
            return round(th, 1)
        self.report.append("阈值未自动更新（样本不足或两组无分离）")
        return None

    def train(self, model_path: str, cfg: Optional[DetectConfig] = None) -> Model:
        """训练并写模型文件。传入 cfg 时会就地更新推导出的阈值/ROI。"""
        target_cfg = cfg if cfg is not None else self.cfg
        self.cfg = target_cfg
        model = Model()
        model.base_dir = os.path.dirname(os.path.abspath(model_path))
        model.source_images = {k: [p for p, _ in v] for k, v in self.images.items()}

        # 1) ROI：优先用手工配置；否则尝试同视野差分自动定位
        manual_a = list(target_cfg.a_roi)
        manual_ring = list(target_cfg.ring_roi) if hasattr(target_cfg, "ring_roi") else None
        a_ready = manual_a != [0.0, 0.0, 1.0, 1.0]
        if a_ready:
            model.a_roi = manual_a
            self.report.append(f"使用手工工件A ROI：{manual_a}")
        else:
            pair_a = self._first_matching_pair("present", "a")
            if pair_a:
                roi_a = largest_diff_roi(pair_a[0], pair_a[1])
                if roi_a:
                    model.a_roi = roi_a
                    a_ready = True
                    self.report.append(f"自动定位工件A ROI：{roi_a}")
            if not a_ready:
                self.report.append("工件A ROI 未定位：请手工配置 a_roi / a_hole_roi / a_edge_roi，本次不生成A模板")

        ring_ready = bool(manual_ring and manual_ring != [0.0, 0.0, 1.0, 1.0])
        if ring_ready:
            model.ring_roi = manual_ring
            self.report.append(f"使用手工环 ROI：{manual_ring}")
        else:
            pair_ring = self._first_matching_pair("absent", "present")
            if pair_ring:
                roi_ring = largest_diff_roi(pair_ring[0], pair_ring[1])
                if roi_ring:
                    model.ring_roi = roi_ring
                    ring_ready = True
                    self.report.append(f"自动定位环 ROI：{roi_ring}")

        # 2) 自动阈值
        th = self._derive_threshold()
        if th is not None:
            target_cfg.valley_th = th

        # 3) 保存模板
        model.templates["present"] = self._template_set("present", model.ring_roi, model_path)
        model.templates["absent"] = self._template_set("absent", model.ring_roi, model_path)
        model.templates["a"] = self._template_set("a", model.a_roi, model_path) if a_ready else []
        model.stats = {
            "n_absent": len(self.images.get("absent", [])),
            "n_present": len(self.images.get("present", [])),
            "n_a": len(self.images.get("a", [])),
            "valley_th": float(target_cfg.valley_th),
        }
        model.save(model_path)
        self.report.append(f"模型已保存：{model_path}")
        self.report.append(
            "模板数：present=%d absent=%d a=%d" % (
                len(model.templates["present"]), len(model.templates["absent"]),
                len(model.templates["a"])))
        return model
