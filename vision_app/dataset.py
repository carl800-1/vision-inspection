# -*- coding: utf-8 -*-
"""训练照片目录管理。

推荐目录结构（可手工把照片放进去，也可用 GUI 导入）：
    dataset/
        state1_no_ring/    无环
        state2_with_ring/  有环
        state3_workpiece_a/ 放工件A
"""
from __future__ import annotations

import os
from typing import Dict, List

from .imageio import list_images

FOLDER_LABELS = {
    "state1_no_ring": "absent",
    "1_no_ring": "absent",
    "absent": "absent",
    "state2_with_ring": "present",
    "2_with_ring": "present",
    "present": "present",
    "state3_workpiece_a": "a",
    "3_workpiece_a": "a",
    "a": "a",
}


class DatasetManager:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        for name in ("state1_no_ring", "state2_with_ring", "state3_workpiece_a"):
            os.makedirs(os.path.join(root, name), exist_ok=True)

    def scan(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {"absent": [], "present": [], "a": []}
        for folder, label in FOLDER_LABELS.items():
            path = os.path.join(self.root, folder)
            if os.path.isdir(path):
                out[label].extend(list_images(path, recursive=True))
        for k in out:
            out[k] = sorted(set(out[k]))
        return out

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.scan().items()}

    def import_files(self, files: List[str], label: str, copy: bool = True) -> int:
        """把外部照片导入标准目录。label 用 absent/present/a。"""
        import shutil
        folder = {
            "absent": "state1_no_ring",
            "present": "state2_with_ring",
            "a": "state3_workpiece_a",
        }.get(label)
        if not folder:
            return 0
        dest_dir = os.path.join(self.root, folder)
        os.makedirs(dest_dir, exist_ok=True)
        n = 0
        for src in files:
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dest_dir, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                n += 1
                continue
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            n += 1
        return n
