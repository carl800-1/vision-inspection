# -*- coding: utf-8 -*-
"""相机后端：海康 MVS / 演示图 / 本地图片序列。统一输出 BGR ndarray。"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import cv2
import numpy as np

from .config import CameraConfig
from .imageio import imread_unicode, list_images


def prepare_mvs_dll_path() -> None:
    """把 MVS 运行时 DLL 目录加入搜索路径（实测 DLL 在 Common Files）。"""
    common = os.environ.get("CommonProgramFiles(x86)",
                            r"C:\Program Files (x86)\Common Files")
    sdk = os.environ.get("MVCAM_SDK_PATH", r"C:\Program Files (x86)\MVS")
    for d in (
        os.path.join(common, "MVS", "Runtime", "Win64_x64"),
        os.path.join(common, "MVS", "Runtime", "Win32_i86"),
        os.path.join(sdk, "Runtime", "x64"),
        os.path.join(sdk, "Runtime", "Win64_x64"),
        os.path.join(sdk, "Development", "Bin", "Win64"),
        os.path.join(sdk, "Development", "Samples", "Python", "MvImport"),
    ):
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass


class CameraBackend:
    name = "base"

    def enumerate(self) -> List[str]:
        return []

    def connect(self, device_id: str) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def configure(self, cfg: CameraConfig) -> None:
        pass

    def grab(self) -> Optional[np.ndarray]:
        raise NotImplementedError


class DemoBackend(CameraBackend):
    """无相机演示：循环播放两张特写图。"""
    name = "DEMO"

    def __init__(self, demo_images: dict, source_key: str = "交替循环",
                 repeat: int = 5, interval_ms: int = 120):
        self.demo_images = demo_images
        self.source_key = source_key
        self.repeat = repeat
        self.interval_ms = interval_ms
        self._idx = 0
        self._repeat = 0
        self._cache = {}
        for k, p in demo_images.items():
            img = imread_unicode(p)
            if img is not None:
                self._cache[k] = img

    def enumerate(self) -> List[str]:
        return list(self.demo_images.keys()) + ["交替循环"]

    def connect(self, device_id: str) -> bool:
        if not self._cache:
            raise RuntimeError("演示图片读取失败，请检查路径")
        self.source_key = device_id or "交替循环"
        return True

    def grab(self):
        keys = list(self._cache.keys())
        if not keys:
            return None
        if self.source_key == "交替循环":
            img = self._cache[keys[self._idx % len(keys)]]
            self._repeat += 1
            if self._repeat >= self.repeat:
                self._repeat = 0
                self._idx += 1
        else:
            img = self._cache.get(self.source_key, self._cache[keys[0]])
        if self.interval_ms:
            time.sleep(self.interval_ms / 1000.0)
        return img.copy()


class FileSequenceBackend(CameraBackend):
    """把已拍好的照片当视频源，便于离线复现/训练验证。"""
    name = "FILES"

    def __init__(self, folder: str = "", loop: bool = True, interval_ms: int = 150):
        self.folder = folder
        self.loop = loop
        self.interval_ms = interval_ms
        self.files: List[str] = []
        self._idx = 0

    def enumerate(self) -> List[str]:
        return [self.folder] if self.folder else []

    def connect(self, device_id: str) -> bool:
        self.folder = device_id or self.folder
        self.files = list_images(self.folder, recursive=True)
        if not self.files:
            raise RuntimeError(f"目录中没有图片：{self.folder}")
        self._idx = 0
        return True

    def grab(self):
        if not self.files:
            return None
        if self._idx >= len(self.files):
            if not self.loop:
                return None
            self._idx = 0
        img = imread_unicode(self.files[self._idx])
        self._idx += 1
        if self.interval_ms:
            time.sleep(self.interval_ms / 1000.0)
        return img


class HikCameraBackend(CameraBackend):
    """海康 MVS 工业相机（MV-CA050-10GC 等 GigE 型号）。"""
    name = "HIK"

    def __init__(self):
        self.cam = None
        self._opened = False

    def enumerate(self) -> List[str]:
        prepare_mvs_dll_path()
        from hik_camera import HikCamera
        # 不用 get_all_ips()：onefile 打包后会拉起自身导致递归枚举
        return list(HikCamera._get_dev_info().keys())

    def connect(self, device_id: str) -> bool:
        prepare_mvs_dll_path()
        from hik_camera import HikCamera
        self.cam = HikCamera(device_id)
        self.cam.__enter__()
        self._opened = True
        return True

    def disconnect(self) -> None:
        if self._opened and self.cam is not None:
            try:
                self.cam.__exit__(None, None, None)
            except Exception:
                pass
        self._opened, self.cam = False, None

    def configure(self, cfg: CameraConfig) -> None:
        if not self._opened:
            return
        self.cam["ExposureAuto"] = "Off" if not cfg.exposure_auto else "Continuous"
        self.cam["ExposureTime"] = int(cfg.exposure_us * 1000)
        self.cam["Gain"] = float(cfg.gain)

    def grab(self):
        if not self._opened:
            return None
        rgb = self.cam.robust_get_frame()
        if rgb is None:
            return None
        return rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_backend(cfg: CameraConfig, demo_images: dict) -> CameraBackend:
    source = (cfg.source or "demo").lower()
    if source in ("hik", "camera", "mv"):
        return HikCameraBackend()
    if source in ("files", "file", "folder"):
        return FileSequenceBackend(cfg.file_dir, cfg.file_loop)
    return DemoBackend(demo_images)
