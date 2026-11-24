# -*- coding: utf-8 -*-
"""环形轴承防错检测系统 v2 — 几何法为主、示教可选、事件驱动状态机。

运行：
    python ring_inspector_v2.py
    python ring_inspector_v2.py --selftest

源码目录：本文件所在目录；打包后：exe 所在目录。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTextEdit, QCheckBox, QMessageBox, QFrame, QStatusBar, QSizePolicy,
    QFileDialog, QTabWidget,
)

from vision_app.config import AppConfig, load_config, save_config
from vision_app.dataset import DatasetManager
from vision_app.detectors import VisionPipeline
from vision_app.imageio import imread_unicode, imwrite_unicode
from vision_app.model import Model
from vision_app.state_machine import CycleState, Event, PokerStateMachine, StableSignal
from vision_app.trainer import Trainer
from vision_app.camera import create_backend, HikCameraBackend, DemoBackend


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(rel):
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", app_dir()), rel)
    return os.path.join(app_dir(), rel)


HERE = app_dir()
CONFIG_PATH = os.path.join(HERE, "config.json")
DEMO_IMAGES = {
    "有环特写（已放置）": resource_path(os.path.join("demo_images", "with_ring_closeup.png")),
    "无环特写（未放置）": resource_path(os.path.join("demo_images", "without_ring_closeup.png")),
}


def bgr_to_qimage(bgr):
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


class GrabWorker(QThread):
    frame_ready = Signal(object)
    result_ready = Signal(object)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, backend, pipeline, fps_limit=0, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.pipeline = pipeline
        self.fps_limit = float(fps_limit or 0)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        self.status.emit("采集中")
        last = 0.0
        while not self._stop:
            try:
                frame = self.backend.grab()
            except Exception as exc:
                self.error.emit(f"取帧失败：{exc}")
                break
            if self._stop:
                break
            if frame is None:
                self.error.emit("取帧超时")
                continue
            self.frame_ready.emit(frame)
            try:
                self.result_ready.emit(self.pipeline.detect(frame))
            except Exception as exc:
                self.error.emit(f"判定异常：{exc}")
            if self.fps_limit > 0:
                import time
                wait = 1.0 / self.fps_limit - (time.time() - last)
                if wait > 0:
                    self.msleep(int(wait * 1000))
                last = time.time()
        self.status.emit("采集已停止")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("环形轴承防错检测系统 v2  ·  几何判定 + A到位结束")
        self.resize(1320, 880)

        self.cfg = load_config(CONFIG_PATH)
        self.model = Model.load(os.path.join(HERE, self.cfg.paths.model))
        self.pipeline = VisionPipeline(self.cfg.detect, self.model)
        self.backend = None
        self.worker = None
        self.connected = False
        self.grabbing = False
        self.last_frame = None
        self.last_result = None

        self.sm = PokerStateMachine(self.cfg.detect.ring_removal_is_fault)
        self.ring_stable = StableSignal(self.cfg.detect.ring_confirm_frames)
        self.a_stable = StableSignal(self.cfg.detect.a_confirm_frames)
        self.alarm_on = False

        self._build_ui()
        self._load_ui_from_config()
        self._log(f"配置已加载：{CONFIG_PATH}")
        self._log(f"模型：{'已加载 ' + self.cfg.paths.model if self.model else '未找到（几何法可直接运行）'}")

        self.alarm_timer = QTimer(self)
        self.alarm_timer.timeout.connect(self._blink_alarm)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.tabs.addTab(self._build_detect_tab(), "检测")
        self.tabs.addTab(self._build_train_tab(), "照片自学习")
        self.tabs.addTab(self._build_settings_tab(), "参数设置")

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪 · 无相机时可用演示模式或打开图片验证")

    def _build_detect_tab(self):
        page = QWidget()
        main = QHBoxLayout(page)

        left = QVBoxLayout()
        main.addLayout(left, 3)
        self.preview = QLabel("等待图像…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(680, 500)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#1a1a1a;color:#888;border:1px solid #444;")
        left.addWidget(self.preview, 4)

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        rb = QHBoxLayout(frame)
        vf = QFont("Microsoft YaHei", 24, QFont.Bold)
        self.placed_label = QLabel("已放置 ✓")
        self.placed_label.setFont(vf)
        self.placed_label.setAlignment(Qt.AlignCenter)
        self.placed_label.setMinimumHeight(64)
        self.absent_label = QLabel("未放置 ✗")
        self.absent_label.setFont(vf)
        self.absent_label.setAlignment(Qt.AlignCenter)
        self.absent_label.setMinimumHeight(64)
        rb.addWidget(self.placed_label, 1)
        rb.addWidget(self.absent_label, 1)
        info = QVBoxLayout()
        self.state_label = QLabel("流程：状态1 等待环")
        self.state_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.detail_label = QLabel("等待图像…")
        self.detail_label.setFont(QFont("Consolas", 10))
        self.detail_label.setWordWrap(True)
        info.addWidget(self.state_label)
        info.addWidget(self.detail_label)
        rb.addLayout(info, 3)
        left.addWidget(frame, 1)
        self._reset_verdict_style()

        right = QVBoxLayout()
        main.addLayout(right, 1)

        g1 = QGroupBox("相机连接")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)
        self.source_combo.setPlaceholderText("设备 IP / 目录")
        row.addWidget(self.source_combo)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        row.addWidget(self.refresh_btn)
        v1.addLayout(row)
        row2 = QHBoxLayout()
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self._toggle_connect)
        self.demo_btn = QPushButton("演示模式")
        self.demo_btn.setCheckable(True)
        self.demo_btn.setChecked(True)
        self.demo_btn.clicked.connect(self._apply_demo_mode)
        row2.addWidget(self.connect_btn)
        row2.addWidget(self.demo_btn)
        v1.addLayout(row2)
        self.status_label = QLabel("状态：演示模式")
        v1.addWidget(self.status_label)
        right.addWidget(g1)

        g2 = QGroupBox("采集与复位")
        v2 = QVBoxLayout(g2)
        row3 = QHBoxLayout()
        self.single_btn = QPushButton("单帧拍照")
        self.single_btn.clicked.connect(self._single_capture)
        self.cont_btn = QPushButton("连续采集")
        self.cont_btn.setCheckable(True)
        self.cont_btn.clicked.connect(self._toggle_grab)
        row3.addWidget(self.single_btn)
        row3.addWidget(self.cont_btn)
        v2.addLayout(row3)
        row4 = QHBoxLayout()
        self.open_img_btn = QPushButton("打开图片判定")
        self.open_img_btn.clicked.connect(self._open_image)
        self.reset_btn = QPushButton("异常复位")
        self.reset_btn.clicked.connect(self._reset_machine)
        row4.addWidget(self.open_img_btn)
        row4.addWidget(self.reset_btn)
        v2.addLayout(row4)
        right.addWidget(g2)

        g3 = QGroupBox("统计")
        v3 = QVBoxLayout(g3)
        self.count_label = QLabel("OK 0  |  异常 0")
        self.count_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        v3.addWidget(self.count_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        v3.addWidget(self.log)
        right.addWidget(g3)
        return page

    def _build_train_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        g = QGroupBox("照片目录（推荐长期按目录收集）")
        form = QGridLayout(g)
        self.dataset_edit = QComboBox()
        self.dataset_edit.setEditable(True)
        form.addWidget(QLabel("目录"), 0, 0)
        form.addWidget(self.dataset_edit, 0, 1, 1, 3)
        self.dataset_count_label = QLabel("样本：无环 0 / 有环 0 / 工件A 0")
        form.addWidget(self.dataset_count_label, 1, 0, 1, 4)
        v.addWidget(g)

        g2 = QGroupBox("导入照片")
        grid = QGridLayout(g2)
        self.import_btns = {}
        for i, (label, key) in enumerate((
            ("导入 状态1 无环", "absent"),
            ("导入 状态2 有环", "present"),
            ("导入 状态3 工件A", "a"),
        )):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _c, k=key: self._import_images(k))
            self.import_btns[key] = btn
            grid.addWidget(btn, i // 2, i % 2)
        v.addWidget(g2)

        g3 = QGroupBox("训练 / 更新基准")
        v3 = QVBoxLayout(g3)
        row = QHBoxLayout()
        self.train_btn = QPushButton("从目录训练并更新配置")
        self.train_btn.clicked.connect(self._train_from_dataset)
        self.reload_model_btn = QPushButton("重新加载模型")
        self.reload_model_btn.clicked.connect(self._reload_model)
        row.addWidget(self.train_btn)
        row.addWidget(self.reload_model_btn)
        v3.addLayout(row)
        self.train_report = QTextEdit()
        self.train_report.setReadOnly(True)
        self.train_report.setMinimumHeight(240)
        v3.addWidget(self.train_report)
        v.addWidget(g3)
        self._refresh_dataset_counts()
        return page

    def _roi_row(self, label, attr, grid, row):
        grid.addWidget(QLabel(label), row, 0)
        edits = []
        for i, name in enumerate(("x", "y", "w", "h")):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.01)
            sp.setDecimals(3)
            sp.setToolTip(f"{label} {name}（归一化 0~1）")
            grid.addWidget(sp, row, 1 + i)
            edits.append(sp)
        self.roi_edits[attr] = edits

    def _build_settings_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        self.roi_edits = {}

        g1 = QGroupBox("相机")
        f1 = QGridLayout(g1)
        f1.addWidget(QLabel("图源"), 0, 0)
        self.cfg_source = QComboBox()
        self.cfg_source.addItems(["demo", "hik", "files"])
        f1.addWidget(self.cfg_source, 0, 1)
        f1.addWidget(QLabel("设备 / IP"), 1, 0)
        self.cfg_device = QComboBox()
        self.cfg_device.setEditable(True)
        f1.addWidget(self.cfg_device, 1, 1)
        f1.addWidget(QLabel("曝光 μs"), 2, 0)
        self.cfg_exposure = QSpinBox()
        self.cfg_exposure.setRange(1, 1000000)
        f1.addWidget(self.cfg_exposure, 2, 1)
        f1.addWidget(QLabel("增益 dB"), 3, 0)
        self.cfg_gain = QDoubleSpinBox()
        self.cfg_gain.setRange(0, 24)
        f1.addWidget(self.cfg_gain, 3, 1)
        f1.addWidget(QLabel("限帧 fps"), 4, 0)
        self.cfg_fps = QDoubleSpinBox()
        self.cfg_fps.setRange(0, 120)
        f1.addWidget(self.cfg_fps, 4, 1)
        f1.addWidget(QLabel("图片目录"), 5, 0)
        self.cfg_filedir = QComboBox()
        self.cfg_filedir.setEditable(True)
        f1.addWidget(self.cfg_filedir, 5, 1)
        self.cfg_exp_auto = QCheckBox("自动曝光（不建议）")
        f1.addWidget(self.cfg_exp_auto, 6, 0, 1, 2)
        v.addWidget(g1)

        g2 = QGroupBox("判定")
        f2 = QGridLayout(g2)
        f2.addWidget(QLabel("环判定模式"), 0, 0)
        self.cfg_ring_mode = QComboBox()
        self.cfg_ring_mode.addItems(["geometry", "hybrid", "template"])
        f2.addWidget(self.cfg_ring_mode, 0, 1)
        f2.addWidget(QLabel("暗谷阈值"), 1, 0)
        self.cfg_valley_th = QDoubleSpinBox()
        self.cfg_valley_th.setRange(5, 200)
        f2.addWidget(self.cfg_valley_th, 1, 1)
        f2.addWidget(QLabel("置信裕度"), 2, 0)
        self.cfg_margin = QDoubleSpinBox()
        self.cfg_margin.setRange(0, 1)
        self.cfg_margin.setSingleStep(0.05)
        f2.addWidget(self.cfg_margin, 2, 1)
        f2.addWidget(QLabel("A特征模式"), 3, 0)
        self.cfg_a_mode = QComboBox()
        self.cfg_a_mode.addItems(["hybrid", "hole", "edge", "template"])
        f2.addWidget(self.cfg_a_mode, 3, 1)
        f2.addWidget(QLabel("A最少票数"), 4, 0)
        self.cfg_a_votes = QSpinBox()
        self.cfg_a_votes.setRange(1, 3)
        f2.addWidget(self.cfg_a_votes, 4, 1)
        f2.addWidget(QLabel("模板阈值"), 5, 0)
        self.cfg_a_score = QDoubleSpinBox()
        self.cfg_a_score.setRange(0, 1)
        self.cfg_a_score.setSingleStep(0.01)
        f2.addWidget(self.cfg_a_score, 5, 1)
        f2.addWidget(QLabel("中心边缘密度阈值"), 6, 0)
        self.cfg_center_th = QDoubleSpinBox()
        self.cfg_center_th.setRange(0, 1)
        self.cfg_center_th.setSingleStep(0.005)
        self.cfg_center_th.setDecimals(3)
        f2.addWidget(self.cfg_center_th, 6, 1)
        self.cfg_require_center = QCheckBox("状态3要求中心轴与环同时消失")
        f2.addWidget(self.cfg_require_center, 7, 0, 1, 2)
        f2.addWidget(QLabel("环确认帧数"), 8, 0)
        self.cfg_ring_frames = QSpinBox()
        self.cfg_ring_frames.setRange(1, 20)
        f2.addWidget(self.cfg_ring_frames, 8, 1)
        f2.addWidget(QLabel("A确认帧数"), 9, 0)
        self.cfg_a_frames = QSpinBox()
        self.cfg_a_frames.setRange(1, 20)
        f2.addWidget(self.cfg_a_frames, 9, 1)
        self.cfg_ring_fault = QCheckBox("环在A前被取走 → 异常")
        f2.addWidget(self.cfg_ring_fault, 10, 0, 1, 2)
        v.addWidget(g2)

        g3 = QGroupBox("ROI（归一化 0~1，需按固定视野标定）")
        f3 = QGridLayout(g3)
        for i, (label, attr) in enumerate((
            ("环 ROI", "ring_roi"), ("A 模板 ROI", "a_roi"),
            ("A 偏心孔搜索区", "a_hole_roi"), ("A 横边搜索区", "a_edge_roi"),
            ("中心轴/环区", "center_roi"),
        )):
            self._roi_row(label, attr, f3, i)
        v.addWidget(g3)

        row = QHBoxLayout()
        apply_btn = QPushButton("应用参数")
        apply_btn.clicked.connect(self._apply_settings)
        save_btn = QPushButton("保存到 config.json")
        save_btn.clicked.connect(self._save_config_file)
        reload_btn = QPushButton("从文件重新载入")
        reload_btn.clicked.connect(self._reload_config_file)
        row.addWidget(apply_btn)
        row.addWidget(save_btn)
        row.addWidget(reload_btn)
        v.addLayout(row)
        return page

    # ------------------------------------------------------------- 配置读写
    def _load_ui_from_config(self):
        c = self.cfg
        self.cfg_source.setCurrentText(c.camera.source)
        self.cfg_device.setEditText(c.camera.device_id)
        self.cfg_exposure.setValue(c.camera.exposure_us)
        self.cfg_gain.setValue(c.camera.gain)
        self.cfg_fps.setValue(c.camera.fps_limit)
        self.cfg_filedir.setEditText(c.camera.file_dir)
        self.cfg_exp_auto.setChecked(c.camera.exposure_auto)

        d = c.detect
        self.cfg_ring_mode.setCurrentText(d.ring_mode)
        self.cfg_valley_th.setValue(d.valley_th)
        self.cfg_margin.setValue(d.margin_min)
        self.cfg_a_mode.setCurrentText(d.a_feature_mode)
        self.cfg_a_votes.setValue(d.a_min_votes)
        self.cfg_a_score.setValue(d.a_score_th)
        self.cfg_center_th.setValue(d.center_edge_th)
        self.cfg_require_center.setChecked(d.require_center_clear)
        self.cfg_ring_frames.setValue(d.ring_confirm_frames)
        self.cfg_a_frames.setValue(d.a_confirm_frames)
        self.cfg_ring_fault.setChecked(d.ring_removal_is_fault)

        for attr, edits in self.roi_edits.items():
            values = getattr(d, attr)
            for sp, val in zip(edits, values):
                sp.setValue(float(val))

        self.dataset_edit.setEditText(os.path.join(HERE, c.paths.dataset))
        self._refresh_dataset_counts()

    def _sync_config_from_ui(self):
        c = self.cfg
        c.camera.source = self.cfg_source.currentText().strip()
        c.camera.device_id = self.cfg_device.currentText().strip()
        c.camera.exposure_us = self.cfg_exposure.value()
        c.camera.gain = self.cfg_gain.value()
        c.camera.fps_limit = self.cfg_fps.value()
        c.camera.file_dir = self.cfg_filedir.currentText().strip()
        c.camera.exposure_auto = self.cfg_exp_auto.isChecked()

        d = c.detect
        d.ring_mode = self.cfg_ring_mode.currentText()
        d.valley_th = self.cfg_valley_th.value()
        d.margin_min = self.cfg_margin.value()
        d.a_feature_mode = self.cfg_a_mode.currentText()
        d.a_min_votes = self.cfg_a_votes.value()
        d.a_score_th = self.cfg_a_score.value()
        d.center_edge_th = self.cfg_center_th.value()
        d.require_center_clear = self.cfg_require_center.isChecked()
        d.ring_confirm_frames = self.cfg_ring_frames.value()
        d.a_confirm_frames = self.cfg_a_frames.value()
        d.ring_removal_is_fault = self.cfg_ring_fault.isChecked()
        for attr, edits in self.roi_edits.items():
            setattr(d, attr, [sp.value() for sp in edits])

    def _apply_settings(self):
        self._sync_config_from_ui()
        self.ring_stable.confirm = self.cfg.detect.ring_confirm_frames
        self.a_stable.confirm = self.cfg.detect.a_confirm_frames
        self.sm.ring_removal_is_fault = self.cfg.detect.ring_removal_is_fault
        self.pipeline.cfg = self.cfg.detect
        self.pipeline.ring.cfg = self.cfg.detect
        self.pipeline.a.cfg = self.cfg.detect
        self._log("参数已应用到检测管线")

    def _save_config_file(self):
        self._sync_config_from_ui()
        save_config(self.cfg, CONFIG_PATH)
        self._log(f"配置已保存：{CONFIG_PATH}")

    def _reload_config_file(self):
        self.cfg = load_config(CONFIG_PATH)
        self._load_ui_from_config()
        self._apply_settings()
        self._reload_model()
        self._log("配置已从文件重新载入")

    # ------------------------------------------------------------ 日志/样式
    def _log(self, text):
        self.log.append(f"[{datetime.now():%H:%M:%S}] {text}")
        self.train_report.append(f"[{datetime.now():%H:%M:%S}] {text}")

    def _set_status(self, text, color="#666"):
        self.status_label.setText(f"状态：{text}")
        self.status_label.setStyleSheet(f"color:{color};font-weight:bold;")

    _STYLE_ON = "color:{fg};background:{bg};border:2px solid {fg};border-radius:8px;padding:6px;"
    _STYLE_OFF = "color:#BBB;background:#F5F5F5;border:2px solid #E0E0E0;border-radius:8px;padding:6px;"

    def _reset_verdict_style(self):
        self.placed_label.setStyleSheet(self._STYLE_OFF)
        self.absent_label.setStyleSheet(self._STYLE_OFF)

    # -------------------------------------------------------------- 相机控制
    def _refresh_devices(self):
        source = self.cfg_source.currentText()
        try:
            if source == "hik":
                devices = HikCameraBackend().enumerate()
                self.cfg_device.clear()
                self.cfg_device.addItems(devices or ["未发现相机"])
                self.source_combo.clear()
                self.source_combo.addItems(devices or ["未发现相机"])
                self._log(f"枚举海康相机：{len(devices)} 台")
            elif source == "files":
                self.source_combo.clear()
                self.source_combo.addItem(self.cfg.camera.file_dir or "未设置图片目录")
                self._log("已切换到本地图片序列")
            else:
                self._log("当前图源无需枚举设备")
        except Exception as exc:
            self._log(f"枚举失败：{exc}")
            QMessageBox.warning(self, "枚举失败", str(exc))

    def _apply_demo_mode(self, checked=None):
        if self.connected or self.grabbing:
            self._stop_grab()
            self._disconnect()
        self.demo_btn.setChecked(True)
        self.cfg_source.setCurrentText("demo")
        self.source_combo.clear()
        self.source_combo.addItems(list(DEMO_IMAGES.keys()) + ["交替循环"])
        self.source_combo.setCurrentText("交替循环")
        self._set_status("演示模式", "#185FA5")
        self._log("已切换演示模式")

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        if self.demo_btn.isChecked():
            self._log("演示模式请直接点连续采集")
            return
        device = self.source_combo.currentText().strip()
        try:
            self.backend = create_backend(self.cfg.camera, DEMO_IMAGES)
            if isinstance(self.backend, DemoBackend):
                self.backend.source_key = device or "交替循环"
            else:
                self.backend.connect(device)
                self.backend.configure(self.cfg.camera)
            self.connected = True
            self.connect_btn.setText("断开")
            self._set_status(f"已连接 {device}", "#0F6E56")
            self._log(f"已连接：{device}")
        except Exception as exc:
            self.backend = None
            QMessageBox.critical(self, "连接失败", str(exc))
            self._log(f"连接失败：{exc}")

    def _disconnect(self):
        self._stop_grab()
        if self.backend:
            try:
                self.backend.disconnect()
            except Exception:
                pass
        self.backend = None
        self.connected = False
        self.connect_btn.setText("连接")
        self._set_status("未连接", "#666")

    def _ensure_backend(self):
        if self.demo_btn.isChecked():
            if not isinstance(self.backend, DemoBackend):
                self.backend = create_backend(self.cfg.camera, DEMO_IMAGES)
                self.backend.source_key = self.source_combo.currentText() or "交替循环"
            return True
        if not self.connected or self.backend is None:
            QMessageBox.information(self, "提示", "请先连接相机")
            return False
        return True

    def _start_grab(self):
        if not self._ensure_backend():
            return
        self.worker = GrabWorker(self.backend, self.pipeline, self.cfg.camera.fps_limit)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.result_ready.connect(self._on_result)
        self.worker.status.connect(lambda s: self.statusbar.showMessage(s))
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.grabbing = True
        self.cont_btn.setText("停止采集")
        self._log("连续采集已启动")

    def _stop_grab(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.worker = None
        self.grabbing = False
        self.cont_btn.setChecked(False)
        self.cont_btn.setText("连续采集")
        self.alarm_timer.stop()

    def _toggle_grab(self, checked=None):
        if self.grabbing:
            self._stop_grab()
            self._log("连续采集已停止")
        else:
            self._start_grab()

    def _single_capture(self):
        if not self._ensure_backend():
            return
        try:
            frame = self.backend.grab()
        except Exception as exc:
            self._log(f"单帧失败：{exc}")
            return
        if frame is None:
            self._log("单帧失败：取帧超时")
            return
        self._on_frame(frame)
        folder = os.path.join(HERE, self.cfg.paths.captures)
        os.makedirs(folder, exist_ok=True)
        name = f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        imwrite_unicode(os.path.join(folder, name), frame)
        self._handle_result(self.pipeline.detect(frame), single=True)
        self._log(f"单帧已保存：{name}")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                              "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "错误", "图片读取失败")
            return
        self._on_frame(img)
        self._handle_result(self.pipeline.detect(img), single=True)
        self._log(f"本地图片判定：{os.path.basename(path)}")

    def _reset_machine(self):
        self.sm.reset()
        self.ring_stable.reset()
        self.a_stable.reset()
        self.alarm_timer.stop()
        self._reset_verdict_style()
        self._set_status("已复位", "#185FA5")
        self._log("状态机已人工复位")

    # -------------------------------------------------------------- 判定处理
    @Slot(object)
    def _on_frame(self, frame):
        self.last_frame = frame
        qimg = bgr_to_qimage(frame)
        if qimg and not self.preview.size().isEmpty():
            self.preview.setPixmap(QPixmap.fromImage(
                qimg.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    @Slot(object)
    def _on_result(self, res):
        self._handle_result(res, single=False)

    def _handle_result(self, res, single=False):
        d = self.cfg.detect
        if single:
            ring_state = res.ring_state
            a_present = res.a_present
        else:
            ring_state, _rc = self.ring_stable.feed(res.ring_state)
            a_present, _ac = self.a_stable.feed(res.a_present)
        old_state = self.sm.state
        state, event, message = self.sm.feed(ring_state, a_present)
        self.last_result = res

        qimg = bgr_to_qimage(res.overlay)
        if qimg and not self.preview.size().isEmpty():
            self.preview.setPixmap(QPixmap.fromImage(
                qimg.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

        if state == CycleState.FAULT:
            self.absent_label.setStyleSheet(self._STYLE_ON.format(fg="#A32D2D", bg="#FCEBEB"))
            self.placed_label.setStyleSheet(self._STYLE_OFF)
            self.alarm_on = False
            self.alarm_timer.start(500)
        else:
            self.alarm_timer.stop()
            if state == CycleState.CYCLE_DONE:
                self.placed_label.setStyleSheet(self._STYLE_ON.format(fg="#0F6E56", bg="#E1F5EE"))
                self.absent_label.setStyleSheet(self._STYLE_OFF)
            elif ring_state == "PLACED":
                self.placed_label.setStyleSheet(self._STYLE_ON.format(fg="#0F6E56", bg="#E1F5EE"))
                self.absent_label.setStyleSheet(self._STYLE_OFF)
            elif ring_state == "ABSENT" and state == CycleState.IDLE:
                self.placed_label.setStyleSheet(self._STYLE_OFF)
                self.absent_label.setStyleSheet(
                    self._STYLE_ON.format(fg="#A32D2D", bg="#FCEBEB"))
            else:
                self._reset_verdict_style()

        self.state_label.setText(
            f"流程：状态{self.sm.state_index} {state.value}  |  环={ring_state}  "
            f"A={'到位' if a_present else '未到'}")
        self.detail_label.setText(message + "\n" + (res.notes or "") + "\n" +
            f"暗谷 {res.geometry.get('valley')}  阈值 {d.valley_th}  "
            f"中心边缘 {res.center_edge:.3f}（清空={res.center_clear}）")
        self.count_label.setText(f"OK {self.sm.ok_count}  |  异常 {self.sm.fault_count}")

        if event != Event.NONE or state != old_state:
            self._log(f"[{state.value}] {event.value} {message}")
            if d.save_event_frames and self.last_frame is not None:
                folder = os.path.join(HERE, self.cfg.paths.events)
                os.makedirs(folder, exist_ok=True)
                name = f"{datetime.now():%Y%m%d_%H%M%S}_{state.value}_{event.value}.jpg"
                imwrite_unicode(os.path.join(folder, name), self.last_frame)
        if state == CycleState.FAULT:
            self._set_status(message, "#A32D2D")
        elif state == CycleState.CYCLE_DONE:
            self._set_status("本件 OK", "#0F6E56")
        else:
            self._set_status(message, "#185FA5")

    def _blink_alarm(self):
        self.alarm_on = not self.alarm_on
        bg = "#F7C1C1" if self.alarm_on else "#FCEBEB"
        self.placed_label.setStyleSheet(self._STYLE_OFF)
        self.absent_label.setStyleSheet(self._STYLE_ON.format(fg="#A32D2D", bg=bg))

    @Slot(str)
    def _on_error(self, msg):
        self._log(f"异常：{msg}")
        self._set_status("相机异常", "#A32D2D")

    # -------------------------------------------------------------- 自学习区
    def _dataset_root(self):
        root = self.dataset_edit.currentText().strip() or os.path.join(HERE, self.cfg.paths.dataset)
        os.makedirs(root, exist_ok=True)
        return root

    def _refresh_dataset_counts(self):
        try:
            counts = DatasetManager(self._dataset_root()).counts()
        except Exception:
            counts = {"absent": 0, "present": 0, "a": 0}
        self.dataset_count_label.setText(
            f"样本：无环 {counts['absent']} / 有环 {counts['present']} / 工件A {counts['a']}")

    def _import_images(self, label):
        files, _ = QFileDialog.getOpenFileNames(self, "选择照片（可多选）", "",
                                                "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        n = DatasetManager(self._dataset_root()).import_files(files, label)
        self._refresh_dataset_counts()
        self._log(f"导入 {label}：{n} 张")

    def _train_from_dataset(self):
        root = self._dataset_root()
        scanned = DatasetManager(root).scan()
        trainer = Trainer(self.cfg.detect)
        total = 0
        for label, paths in scanned.items():
            for p in paths:
                total += int(trainer.add_image(p, label))
        if scanned["absent"] == 0 or scanned["present"] == 0:
            QMessageBox.information(self, "样本不足", "至少各需要一张状态1无环、状态2有环照片")
            return
        model_path = os.path.join(HERE, self.cfg.paths.model)
        self._sync_config_from_ui()
        model = trainer.train(model_path, self.cfg.detect)
        save_config(self.cfg, CONFIG_PATH)
        self._reload_model()
        self.train_report.append("--- 训练报告 ---")
        for line in trainer.report:
            self.train_report.append(line)
        self._log(f"训练完成：{total} 张照片，模型 {model_path}")
        QMessageBox.information(self, "完成", f"已训练 {total} 张照片\n" + "\n".join(trainer.report[-4:]))

    def _reload_model(self):
        self.model = Model.load(os.path.join(HERE, self.cfg.paths.model))
        self.pipeline.set_model(self.model)
        self._log(f"模型重载：{'成功' if self.model else '未找到，使用纯几何法'}")

    def closeEvent(self, event):
        self._stop_grab()
        if self.backend:
            self.backend.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    icon = resource_path("app_icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    win = MainWindow()
    win.show()

    if "--selftest" in sys.argv:
        win._apply_demo_mode()
        win._start_grab()

        def finish():
            win._stop_grab()
            out = app_dir()
            if win.last_result and win.last_result.overlay is not None:
                imwrite_unicode(os.path.join(out, "selftest_v2.png"), win.last_result.overlay)
            with open(os.path.join(out, "selftest_v2.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"state={win.sm.state.value}\n")
                fh.write(f"ok={win.sm.ok_count} fault={win.sm.fault_count}\n")
                fh.write(win.log.toPlainText()[-1200:])
            app.quit()

        QTimer.singleShot(5000, finish)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
