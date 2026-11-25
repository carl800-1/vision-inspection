# -*- coding: utf-8 -*-
"""示教模型：只保存参数与模板图片路径，便于后续增量训练。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from .imageio import imread_unicode, imwrite_unicode

MODEL_VERSION = "1.1"


@dataclass
class Model:
    version: str = MODEL_VERSION
    created_at: str = ""
    ring_roi: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.6, 0.6])
    a_roi: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    # key: present / absent / a / locator → 模板图片相对路径列表
    templates: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    source_images: Dict[str, List[str]] = field(default_factory=dict)
    base_dir: str = ""
    # 裁剪模板法：环心（训练后验坐标，按训练图尺寸）与定位模板
    ring_center: List[float] = field(default_factory=lambda: [0.0, 0.0])  # 归一化 x,y
    crop_half: int = 130                # 裁剪半径（在 max_side 尺度下的像素）
    locator_roi: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    image_size: List[int] = field(default_factory=lambda: [0, 0])  # 训练图工作尺寸 w,h

    @property
    def has_locator(self) -> bool:
        return bool(self.templates.get("locator"))

    @property
    def has_ring_template(self) -> bool:
        return bool(self.templates.get("present")) and bool(self.templates.get("absent"))

    @property
    def has_a_template(self) -> bool:
        return bool(self.templates.get("a"))

    def template_images(self, key: str):
        """返回该类别的 BGR 模板图列表（读取失败自动跳过）。"""
        out = []
        base = self.base_dir or "."
        for rel in self.templates.get(key, []):
            path = rel if os.path.isabs(rel) else os.path.join(base, rel)
            img = imread_unicode(path)
            if img is not None:
                out.append(img)
        return out

    def save(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.created_at = self.created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "version": self.version,
            "created_at": self.created_at,
            "ring_roi": self.ring_roi,
            "a_roi": self.a_roi,
            "templates": self.templates,
            "stats": self.stats,
            "source_images": self.source_images,
            "ring_center": self.ring_center,
            "crop_half": self.crop_half,
            "locator_roi": self.locator_roi,
            "image_size": self.image_size,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str):
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return None
        m = Model()
        m.base_dir = os.path.dirname(os.path.abspath(path))
        for key in ("version", "created_at", "ring_roi", "a_roi",
                    "templates", "stats", "source_images", "ring_center",
                    "crop_half", "locator_roi", "image_size"):
            if key in data:
                setattr(m, key, data[key])
        return m


def save_template_image(model: Model, model_path: str, key: str, img, index: int = 0) -> str:
    """把模板图写到 model/templates/<key>_<n>.png，返回相对路径。"""
    base = os.path.dirname(os.path.abspath(model_path))
    tdir = os.path.join(base, "templates")
    os.makedirs(tdir, exist_ok=True)
    rel = os.path.join("templates", f"{key}_{index:02d}.png")
    imwrite_unicode(os.path.join(base, rel), img)
    return rel.replace("\\", "/")
