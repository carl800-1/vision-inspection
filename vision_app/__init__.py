# -*- coding: utf-8 -*-
"""环形轴承防错检测 — 可复用核心模块。

模块划分：
    config        参数定义与 JSON 读写
    imageio       中文路径安全的图像读写、ROI 工具
    detectors     环判定 / 工件A判定，支持几何法 + 示教模板
    trainer       照片导入、自动定位ROI、训练并保存模型
    state_machine 1→2→3 事件驱动防错状态机
    camera        相机后端（海康 / 演示 / 本地图片序列）
    dataset       训练照片目录管理
    demo_flow     照片流程演示（顺序入库、逐张判定、记录与报告）
"""
from .config import (AppConfig, CameraConfig, DetectConfig, DemoFlowConfig,
                     load_config, save_config)
from .imageio import imread_unicode, imwrite_unicode
from .model import Model
from .detectors import DetectionResult, RingDetector, ADetector, VisionPipeline
from .trainer import Trainer
from .state_machine import CycleState, PokerStateMachine
from .camera import DemoBackend, HikCameraBackend, FileSequenceBackend, create_backend
from .dataset import DatasetManager
from .demo_flow import DemoFlow, FlowRecord, PASS, FAIL, PENDING

__all__ = [
    "AppConfig", "CameraConfig", "DetectConfig", "DemoFlowConfig",
    "load_config", "save_config",
    "imread_unicode", "imwrite_unicode",
    "DetectionResult", "RingDetector", "ADetector", "VisionPipeline",
    "Trainer", "Model", "CycleState", "PokerStateMachine",
    "DemoBackend", "HikCameraBackend", "FileSequenceBackend", "create_backend",
    "DatasetManager", "DemoFlow", "FlowRecord", "PASS", "FAIL", "PENDING",
]
