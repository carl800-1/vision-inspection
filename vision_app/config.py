# -*- coding: utf-8 -*-
"""所有可调参数集中在这里；支持从 JSON 覆盖并回写。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict


@dataclass
class CameraConfig:
    """相机接入参数。source: demo | hik | files。"""
    source: str = "demo"
    device_id: str = ""                 # 海康 IP 或设备序号
    exposure_auto: bool = False         # 建议固定曝光，避免判定基准漂移
    exposure_us: int = 20000            # 曝光时间（微秒）
    gain: float = 0.0                   # 增益（dB），建议 0
    fps_limit: float = 8.0              # 连续采集最大频率，0 表示不限
    file_dir: str = ""                  # source=files 时的图片目录
    file_loop: bool = True              # 图片序列是否循环播放


@dataclass
class DetectConfig:
    """检测与防错参数。"""
    # 环判定模式：
    #   crop_template = 先自动定位环心→裁固定小图→模板NCC（新方案，推荐）
    #   geometry      = 径向亮度剖面（旧方案，圆心易错，已弃用）
    #   template      = 直接在归一化ROI上匹配（旧方案）
    #   hybrid        = 旧方案的模板优先
    ring_mode: str = "crop_template"

    # --- 裁剪模板法（新方案）参数 ---
    use_locator: bool = False          # 是否启用运行时环心微调搜索；相机固定时关闭最稳
    locator_th: float = 0.45           # 环定位模板全图匹配的最低 NCC 分数
    crop_half: int = 130               # 以环心为中心的裁剪半径（在 max_side 尺度下）
    ring_ncc_th: float = 0.62          # 有环模板 NCC 阈值：高于它判 PLACED
    a_ncc_th: float = 0.55             # 工件A模板 NCC 阈值：高于它判 A 到位
    locator_scale_search: float = 0.10 # 定位模板尺度搜索范围
    ring_center: list = field(default_factory=lambda: [0.0, 0.0])  # 环心种子(归一化x,y)，[0,0]=训练时自动定位

    # --- 几何法参数（无模型时也能工作） ---
    max_side: int = 1200                # 大图先等比缩到该尺寸再检测，0 表示不缩放
    roi_ratio: float = 0.60             # 中心 ROI 占短边比例
    valley_th: float = 55.0             # 暗谷亮度阈值：<阈值判有环
    margin_min: float = 0.05            # 距阈值相对裕度下限，过小判 UNCERTAIN
    hough_param2: int = 45              # 霍夫圆累加器阈值
    radius_lo: float = 0.18             # 霍夫半径下限（短边比例）
    radius_hi: float = 0.42             # 霍夫半径上限（短边比例）
    band_lo: float = 0.30               # 剖面搜索带下限（r0 倍数）
    band_hi: float = 1.25               # 剖面搜索带上限（r0 倍数）

    # --- 示教模板参数（导入照片训练后生效） ---
    ring_roi: list = field(default_factory=lambda: [0.2, 0.2, 0.6, 0.6])  # 环模板 ROI
    template_th: float = 0.82           # 模板 NCC 相似度阈值
    template_margin: float = 0.05       # 有环模板相似度至少高出无环模板多少
    scale_search: float = 0.06          # 模板尺度搜索 ±比例，0 表示不搜索

    # --- 工件 A 参数 ---
    a_feature_mode: str = "hybrid"      # hole=偏心孔；edge=横边；template=模板；hybrid=投票
    a_min_votes: int = 2                # hybrid 模式下至少几票才算 A 到位
    a_score_th: float = 0.78            # A 模板 NCC 阈值
    a_roi: list = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # A 模板/特征 ROI，归一化 x,y,w,h
    a_hole_roi: list = field(default_factory=lambda: [0.45, 0.05, 0.50, 0.45])  # 找偏心孔的搜索区
    a_hole_param2: int = 22             # 偏心孔霍夫累加器阈值（越小越灵敏）
    a_hole_r_lo: float = 0.03           # 孔半径下限（短边比例）
    a_hole_r_hi: float = 0.16           # 孔半径上限（短边比例）
    a_hole_min_offset: float = 0.12     # 孔心距图像中心最小归一化距离
    a_edge_roi: list = field(default_factory=lambda: [0.40, 0.05, 0.55, 0.55])  # 找横边的搜索区
    a_edge_min_len: int = 60            # 横边最短像素长度，0 表示按短边 12% 自动取
    a_edge_th: int = 80                 # Canny 低阈值
    center_roi: list = field(default_factory=lambda: [0.46, 0.58, 0.08, 0.09])  # 中心轴/环遮挡检测区
    center_edge_th: float = 0.22        # 中心区域边缘密度阈值：低于此值认为轴与环同时消失
    require_center_clear: bool = True   # 状态3必须同时满足“中心轴/环消失”

    # --- 去抖与业务流程 ---
    ring_confirm_frames: int = 3        # 环状态连续确认帧数
    a_confirm_frames: int = 2           # A 出现连续确认帧数
    ring_removal_is_fault: bool = True  # 锁存有环后、A 出现前环消失 → 报警
    save_event_frames: bool = True      # 关键事件保存原始帧到 captures/events


@dataclass
class PathConfig:
    """相对程序目录的路径。"""
    dataset: str = "dataset"
    model: str = "model/ring_model.json"
    captures: str = "captures"
    events: str = "captures/events"
    logs: str = "logs"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    app_version: str = "1.0.2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(target: Any, source: Dict[str, Any]) -> None:
    """把 source 的键递归覆盖到 dataclass 实例 target。"""
    if not isinstance(source, dict):
        return
    for f in fields(target):
        if f.name not in source:
            continue
        value = source[f.name]
        current = getattr(target, f.name)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, f.name, value)


def load_config(path: str) -> AppConfig:
    cfg = AppConfig()
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _merge(cfg, json.load(fh))
        except Exception as exc:
            raise RuntimeError(f"配置文件读取失败：{path} — {exc}") from exc
    return cfg


def save_config(cfg: AppConfig, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=2)
