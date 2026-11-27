# -*- coding: utf-8 -*-
"""环形轴承防错检测系统 v2 — 几何法为主、示教可选、事件驱动状态机。

运行：
    python ring_inspector_v2.py
    python ring_inspector_v2.py --selftest

源码目录：本文件所在目录；打包后：exe 所在目录。
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPointF, QRectF
from PySide6.QtGui import (QImage, QPixmap, QFont, QIcon, QColor, QBrush,
                           QPainter, QPen, QFontMetrics)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTextEdit, QCheckBox, QMessageBox, QFrame, QStatusBar, QSizePolicy,
    QFileDialog, QTabWidget, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QProgressBar, QHeaderView, QAbstractItemView, QScrollArea,
)

from vision_app.auto_flow import AutoFlowController, AutoEvent, AutoState
from vision_app.config import AppConfig, load_config, save_config
from vision_app.demo_flow import DemoFlow, PASS, FAIL
from vision_app.detectors import VisionPipeline
from vision_app.imageio import imread_unicode, imwrite_unicode
from vision_app.model import Model
from vision_app.state_machine import CycleState, Event, PokerStateMachine, StableSignal
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


class AutoShotWorker(QThread):
    """自动流程用：每次启动只拍一张、判一张，成功/失败都发信号。"""
    shot_ready = Signal(object, object)   # (frame, DetectionResult)
    failed = Signal(str, str)             # (阶段, 原因)

    def __init__(self, backend, pipeline, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.pipeline = pipeline

    def run(self):
        try:
            frame = self.backend.grab()
        except Exception as exc:
            self.failed.emit("拍照失败", str(exc))
            return
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            self.failed.emit("拍照失败", "取帧为空或超时")
            return
        try:
            res = self.pipeline.detect(frame)
        except Exception as exc:
            self.failed.emit("检测失败", str(exc))
            return
        if res is None:
            self.failed.emit("检测失败", "检测结果为空")
            return
        self.shot_ready.emit(frame, res)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("环形轴承防错检测系统 v1.0.7  ·  裁剪模板判定 + 自动拍照 / 流程演示")
        self.resize(1320, 960)

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

        # 相机自动拍照流程：独立控制器，复用同一检测管线
        self.auto = AutoFlowController(self.cfg.auto_flow)
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)
        self.auto_worker = None
        self.auto_running = False
        self.auto_alarm_timer = QTimer(self)
        self.auto_alarm_timer.timeout.connect(self._auto_blink_alarm)

        # 照片流程演示：独立状态机，复用检测管线
        self.flow = DemoFlow(self.cfg.demo_flow, self.cfg.detect, HERE, self.model)
        self.flow_timer = QTimer(self)
        self.flow_timer.timeout.connect(self._flow_auto_step)

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
        self.tabs.addTab(self._build_flow_tab(), "流程演示")
        # 参数设置页内容多（最小高度约1046px），用滚动容器包住，避免把整个窗口
        # 最小高度顶到屏幕外，导致检测页左下角判定条在1080p屏幕上被挤出视野
        settings_scroll = QScrollArea()
        settings_scroll.setWidget(self._build_settings_tab())
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        self.tabs.addTab(settings_scroll, "参数设置")

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
        self.preview.setMinimumSize(680, 420)
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
        self.state_label = QLabel("流程：安装结束/待机（等待检测到轴）")
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

        g_auto = QGroupBox("自动拍照检测")
        va = QVBoxLayout(g_auto)
        # 节点横幅已移到流程演示页（照片序列上方），检测页保持简洁
        self.auto_labels = {}
        self.auto_detail_label = QLabel(
            "每间隔自动拍一张，首次「有主轴、无轴承」照片启动流程。参数见参数设置页。")
        self.auto_detail_label.setWordWrap(True)
        self.auto_detail_label.setFont(QFont("Consolas", 9))
        va.addWidget(self.auto_detail_label)
        rowa = QHBoxLayout()
        self.auto_start_btn = QPushButton("启动自动拍照")
        self.auto_start_btn.setStyleSheet("color:#0F6E56;background:#E1F5EE;font-weight:bold;")
        self.auto_start_btn.clicked.connect(self._auto_toggle)
        self.auto_reset_btn = QPushButton("复位")
        self.auto_reset_btn.clicked.connect(self._auto_reset)
        self.auto_mute = QCheckBox("报警静音")
        self.auto_export_btn = QPushButton("导出记录 CSV")
        self.auto_export_btn.clicked.connect(self._auto_export)
        rowa.addWidget(self.auto_start_btn)
        rowa.addWidget(self.auto_reset_btn)
        rowa.addWidget(self.auto_mute)
        rowa.addWidget(self.auto_export_btn)
        va.addLayout(rowa)
        self.auto_table = QTableWidget()
        self.auto_table.setColumnCount(6)
        self.auto_table.setHorizontalHeaderLabels(
            ["序号", "拍照", "流程节点", "显示", "说明", "报警"])
        self.auto_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.auto_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.auto_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.auto_table.setMaximumHeight(160)
        va.addWidget(self.auto_table)
        right.addWidget(g_auto)

        g3 = QGroupBox("统计")
        v3 = QVBoxLayout(g3)
        self.count_label = QLabel("OK 0  |  异常 0")
        self.count_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        v3.addWidget(self.count_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        v3.addWidget(self.log)
        right.addWidget(g3)
        return page

    # ---------------------------------------------------------- 流程演示页
    def _build_flow_tab(self):
        """流程演示：选一个照片文件夹，按文件名顺序逐张读取、判定所属流程节点并显示，核对判定是否有错。与相机无关。"""
        page = QWidget()
        main = QHBoxLayout(page)

        # --- 左：预览 + 序列 + 进度 ---
        left = QVBoxLayout()
        main.addLayout(left, 3)

        self.flow_preview = QLabel("准备照片后将显示当前张")
        self.flow_preview.setAlignment(Qt.AlignCenter)
        self.flow_preview.setMinimumSize(560, 320)
        self.flow_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.flow_preview.setStyleSheet("background:#1a1a1a;color:#888;border:1px solid #444;")
        left.addWidget(self.flow_preview, 3)

        seq_box = QGroupBox("照片序列（按文件名顺序，当前张高亮）")
        seq_v = QVBoxLayout(seq_box)
        # 流程节点横幅：与检测页一致，随每张照片判定结果高亮当前节点（带图标/箭头/呼吸动画）
        banner = QHBoxLayout()
        banner.setSpacing(2)
        self.flow_node_labels = {}
        node_defs = (("idle", "● 安装结束/待机"), ("no_bearing", "▶ 无轴承"),
                     ("placed", "◐ 放置轴承"), ("missing", "⚠ 漏装"))
        for i, (key, text) in enumerate(node_defs):
            if i:
                arrow = QLabel("⚠" if key == "missing" else "→")
                arrow.setStyleSheet(f"color:{'#A32D2D' if key == 'missing' else '#999'};"
                                    "font-size:15px;font-weight:bold;padding:0 2px;")
                arrow.setAlignment(Qt.AlignCenter)
                banner.addWidget(arrow)
            lab = QLabel(text)
            lab.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumHeight(36)
            self.flow_node_labels[key] = lab
            banner.addWidget(lab, 1)
        seq_v.addLayout(banner)
        self._banner_active = None
        self._banner_pulse = True
        self._banner_timer = QTimer(self)
        self._banner_timer.setInterval(700)
        self._banner_timer.timeout.connect(self._banner_pulse_tick)
        self._flow_render_banner()
        # 预览图左上角的判定结果徽章（随每张照片更新）
        self.flow_badge = QLabel(self.flow_preview)
        self.flow_badge.hide()
        self.flow_seq_list = QListWidget()
        self.flow_seq_list.setMaximumHeight(100)
        self.flow_seq_list.itemClicked.connect(self._flow_goto)
        seq_v.addWidget(self.flow_seq_list)
        left.addWidget(seq_box, 1)

        prog_box = QGroupBox("处理进度")
        prog_v = QVBoxLayout(prog_box)
        self.flow_progress = QProgressBar()
        self.flow_progress.setRange(0, 100)
        self.flow_progress.setValue(0)
        prog_v.addWidget(self.flow_progress)
        self.flow_progress_label = QLabel("0 / 0 张")
        prog_v.addWidget(self.flow_progress_label)
        left.addWidget(prog_box)

        # --- 右：配置 + 控制 + 记录 ---
        right = QVBoxLayout()
        main.addLayout(right, 2)

        cfg_box = QGroupBox("演示配置")
        fg = QGridLayout(cfg_box)
        fg.addWidget(QLabel("照片目录"), 0, 0)
        self.flow_src = QComboBox()
        self.flow_src.setEditable(True)
        fg.addWidget(self.flow_src, 0, 1, 1, 3)
        self.flow_src_btn = QPushButton("浏览")
        self.flow_src_btn.clicked.connect(self._flow_browse_source)
        fg.addWidget(self.flow_src_btn, 0, 4)

        fg.addWidget(QLabel("目标文件夹"), 1, 0)
        self.flow_dst = QComboBox()
        self.flow_dst.setEditable(True)
        self.flow_dst.setToolTip("照片按顺序复制进去，保证每次演示顺序一致、可重复")
        fg.addWidget(self.flow_dst, 1, 1, 1, 3)
        self.flow_dst_btn = QPushButton("浏览")
        self.flow_dst_btn.clicked.connect(self._flow_browse_target)
        fg.addWidget(self.flow_dst_btn, 1, 4)

        fg.addWidget(QLabel("播放间隔 ms"), 2, 0)
        self.flow_interval = QSpinBox()
        self.flow_interval.setRange(200, 10000)
        self.flow_interval.setSingleStep(100)
        fg.addWidget(self.flow_interval, 2, 1)
        self.flow_loop = QCheckBox("循环")
        fg.addWidget(self.flow_loop, 2, 2)

        self.flow_clear = QCheckBox("准备时清空目标文件夹（保证可重复）")
        self.flow_clear.setChecked(True)
        fg.addWidget(self.flow_clear, 3, 0, 1, 5)
        right.addWidget(cfg_box)

        ctrl_box = QGroupBox("流程控制")
        cv = QVBoxLayout(ctrl_box)
        row_a = QHBoxLayout()
        self.flow_prepare_btn = QPushButton("准备照片（按顺序放入文件夹）")
        self.flow_prepare_btn.clicked.connect(self._flow_prepare)
        row_a.addWidget(self.flow_prepare_btn)
        self.flow_reset_btn = QPushButton("重置重跑")
        self.flow_reset_btn.clicked.connect(self._flow_reset)
        row_a.addWidget(self.flow_reset_btn)
        cv.addLayout(row_a)

        row_b = QHBoxLayout()
        self.flow_prev_btn = QPushButton("上一张")
        self.flow_prev_btn.clicked.connect(self._flow_prev)
        self.flow_next_btn = QPushButton("下一张（判定）")
        self.flow_next_btn.clicked.connect(self._flow_next)
        self.flow_auto_btn = QPushButton("自动播放")
        self.flow_auto_btn.setCheckable(True)
        self.flow_auto_btn.clicked.connect(self._flow_toggle_auto)
        row_b.addWidget(self.flow_prev_btn)
        row_b.addWidget(self.flow_next_btn)
        row_b.addWidget(self.flow_auto_btn)
        cv.addLayout(row_b)

        self.flow_detail = QLabel("等待处理…")
        self.flow_detail.setWordWrap(True)
        self.flow_detail.setFont(QFont("Consolas", 9))
        cv.addWidget(self.flow_detail)
        right.addWidget(ctrl_box)

        rec_box = QGroupBox("处理记录（每张判定出的流程节点 / 状态变化）")
        rv = QVBoxLayout(rec_box)
        legend = QLabel("预期节点链：安装结束/待机 →（检测到轴）无轴承 →（检测到轴承）放置轴承 →（检测到工件A）安装结束/待机；异常照片 → 报警")
        legend.setStyleSheet("color:#555;")
        legend.setWordWrap(True)
        rv.addWidget(legend)
        self.flow_table = QTableWidget()
        self.flow_table.setColumnCount(5)
        self.flow_table.setHorizontalHeaderLabels(
            ["序号", "文件名", "流程节点", "自动判定", "状态变化/事件"])
        self.flow_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.flow_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.flow_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.flow_table.itemClicked.connect(self._flow_table_clicked)
        self.flow_table.setMinimumHeight(180)
        rv.addWidget(self.flow_table)
        row_d = QHBoxLayout()
        self.flow_save_cfg_btn = QPushButton("保存配置")
        self.flow_save_cfg_btn.clicked.connect(self._flow_save_config)
        self.flow_export_btn = QPushButton("导出报告 CSV")
        self.flow_export_btn.clicked.connect(self._flow_export)
        row_d.addWidget(self.flow_save_cfg_btn)
        row_d.addWidget(self.flow_export_btn)
        rv.addLayout(row_d)
        right.addWidget(rec_box)

        self._load_flow_ui_from_config()
        self._flow_refresh_sequence()
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
        self.cfg_ring_mode.addItems(["crop_template", "geometry", "hybrid", "template"])
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

        g4 = QGroupBox("自动拍照（流程演示 · 相机模式）")
        f4 = QGridLayout(g4)
        f4.addWidget(QLabel("拍照间隔 ms"), 0, 0)
        self.auto_interval = QSpinBox()
        self.auto_interval.setRange(100, 60000)
        self.auto_interval.setSingleStep(100)
        f4.addWidget(self.auto_interval, 0, 1)
        f4.addWidget(QLabel("有效照片超时 ms"), 1, 0)
        self.auto_timeout = QSpinBox()
        self.auto_timeout.setRange(200, 60000)
        self.auto_timeout.setSingleStep(100)
        f4.addWidget(self.auto_timeout, 1, 1)
        # 「连续有轴承张数」已废弃：完成依据改为「检测到工件A」，无需人工设张数
        f4.addWidget(QLabel("最大无效重试"), 2, 0)
        self.auto_retries = QSpinBox()
        self.auto_retries.setRange(1, 20)
        f4.addWidget(self.auto_retries, 2, 1)
        self.auto_need_shaft = QCheckBox("启动需同时检出主轴")
        f4.addWidget(self.auto_need_shaft, 3, 0, 1, 2)
        self.auto_confident = QCheckBox("环状态 UNCERTAIN 视为检测失败")
        f4.addWidget(self.auto_confident, 4, 0, 1, 2)
        self.auto_alarm_enabled = QCheckBox("漏装时报警（蜂鸣+闪烁）")
        f4.addWidget(self.auto_alarm_enabled, 5, 0, 1, 2)
        self.auto_save = QCheckBox("保存每张自动照片到 captures/auto")
        f4.addWidget(self.auto_save, 6, 0, 1, 2)
        v.addWidget(g4)

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
        self._load_auto_ui_from_config()
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
        self._sync_auto_ui_to_config()

    def _apply_settings(self):
        self._sync_config_from_ui()
        self.ring_stable.confirm = self.cfg.detect.ring_confirm_frames
        self.a_stable.confirm = self.cfg.detect.a_confirm_frames
        self.sm.ring_removal_is_fault = self.cfg.detect.ring_removal_is_fault
        self.pipeline.cfg = self.cfg.detect
        self.pipeline.ring.cfg = self.cfg.detect
        self.pipeline.a.cfg = self.cfg.detect
        # 流程演示复用同一套判定参数
        self.flow.detect_cfg = self.cfg.detect
        self.flow.pipeline.cfg = self.cfg.detect
        self.flow.pipeline.ring.cfg = self.cfg.detect
        self.flow.pipeline.a.cfg = self.cfg.detect
        self.flow.ring_stable.confirm = self.cfg.detect.ring_confirm_frames
        self.flow.a_stable.confirm = self.cfg.detect.a_confirm_frames
        self.flow.sm.ring_removal_is_fault = self.cfg.detect.ring_removal_is_fault
        self.auto.cfg = self.cfg.auto_flow
        self._log("参数已应用到检测管线、自动流程与流程演示")

    def _save_config_file(self):
        self._sync_config_from_ui()
        save_config(self.cfg, CONFIG_PATH)
        self._log(f"配置已保存：{CONFIG_PATH}")

    def _reload_config_file(self):
        self.cfg = load_config(CONFIG_PATH)
        self._load_ui_from_config()
        self._load_auto_ui_from_config()
        self._load_flow_ui_from_config()
        self.flow.cfg = self.cfg.demo_flow
        self.auto.cfg = self.cfg.auto_flow
        self._apply_settings()
        self._reload_model()
        self._log("配置已从文件重新载入")

    # ---------------------------------------------------------- 自动流程配置
    def _load_auto_ui_from_config(self):
        a = self.cfg.auto_flow
        self.auto_interval.setValue(int(a.interval_ms))
        self.auto_timeout.setValue(int(a.photo_timeout_ms))
        self.auto_retries.setValue(int(a.max_invalid_retries))
        self.auto_need_shaft.setChecked(bool(a.require_shaft))
        self.auto_confident.setChecked(bool(a.require_confident))
        self.auto_alarm_enabled.setChecked(bool(a.alarm_enabled))
        self.auto_save.setChecked(bool(a.save_captures))

    def _sync_auto_ui_to_config(self):
        a = self.cfg.auto_flow
        a.interval_ms = int(self.auto_interval.value())
        a.photo_timeout_ms = int(self.auto_timeout.value())
        a.max_invalid_retries = int(self.auto_retries.value())
        a.require_shaft = bool(self.auto_need_shaft.isChecked())
        a.require_confident = bool(self.auto_confident.isChecked())
        a.alarm_enabled = bool(self.auto_alarm_enabled.isChecked())
        a.save_captures = bool(self.auto_save.isChecked())
        self.auto.cfg = a

    # ------------------------------------------------------------ 日志/样式
    def _log(self, text):
        self.log.append(f"[{datetime.now():%H:%M:%S}] {text}")

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
        if self.auto_running:
            self._auto_stop("切换演示模式，自动流程停止")
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
        if self.auto_running:
            self._auto_stop("切换相机连接，自动流程停止")
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
        if self.auto_running:
            self._auto_stop("相机断开，自动流程停止")
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
        if self.auto_running:
            self._auto_stop("切换到连续采集，自动流程停止")
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
            if event == Event.CYCLE_OK or (state == CycleState.IDLE and a_present):
                # 安装结束/待机（合并态）：本件完成或A在位都点亮绿色
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
        elif event == Event.CYCLE_OK:
            self._set_status("本件 OK（安装结束/待机）", "#0F6E56")
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

    # ------------------------------------------------ 自动拍照（流程演示·相机模式）
    def _auto_toggle(self):
        if self.auto_running:
            self._auto_stop("自动拍照已停止")
        else:
            self._auto_start()

    def _auto_start(self):
        self._sync_auto_ui_to_config()
        if self.grabbing:
            self._stop_grab()
            self._log("连续采集已停止，切换到自动拍照")
        if not self._ensure_backend():
            return
        self.auto.start()
        self.auto_running = True
        self.auto_timer.start(max(100, self.cfg.auto_flow.interval_ms))
        self.auto_start_btn.setText("停止自动拍照")
        self.auto_start_btn.setStyleSheet(
            "color:#A32D2D;background:#FCEBEB;font-weight:bold;")
        self._auto_render_labels(self.auto.state)
        self._auto_refresh_table()
        self.auto_detail_label.setText(
            f"每 {self.cfg.auto_flow.interval_ms}ms 自动拍一张；"
            f"「检测到轴」启动流程，「检测到工件A」结束本件。")
        self._set_status("自动拍照运行中", "#185FA5")
        self._log(f"自动拍照启动：间隔 {self.cfg.auto_flow.interval_ms}ms，"
                  f"超时 {self.cfg.auto_flow.photo_timeout_ms}ms，"
                  f"完成依据：检测到工件A")

    def _auto_stop(self, msg="自动拍照已停止"):
        was_alarm = self.auto.state == AutoState.MISSING
        self.auto_timer.stop()
        if self.auto_worker and self.auto_worker.isRunning():
            # 不阻塞 GUI 等待相机超时，线程自然结束后由 finished 信号回收
            self.auto_worker.finished.connect(self.auto_worker.deleteLater)
        self.auto_running = False
        self.auto.stop()
        self.auto_start_btn.setText("启动自动拍照")
        self.auto_start_btn.setStyleSheet(
            "color:#0F6E56;background:#E1F5EE;font-weight:bold;")
        self._auto_stop_alarm()
        self._auto_render_labels(self.auto.state)
        if msg:
            self._log(msg)
            self._set_status(msg, "#A32D2D" if was_alarm else "#666")

    def _auto_reset(self):
        if self.auto_running:
            self._auto_stop("")
        rec = self.auto.reset()
        self.auto_alarm_timer.stop()
        self._auto_render_labels(self.auto.state)
        self._auto_refresh_table()
        self.auto_detail_label.setText(
            f"{rec.message}（状态 {self.auto.display} ｜ 有效照片 0 ｜ 拍照序号 0）")
        self._log("自动拍照已复位，可重新启动")

    def _auto_tick(self):
        if not self.auto_running:
            return
        if (not self.demo_btn.isChecked()) and (not self.connected or self.backend is None):
            self._auto_stop("相机已断开，自动拍照停止")
            return
        # 优先检查超时：即使拍照一直失败，也要按时给出「漏装」报警
        rec = self.auto.check_timeout()
        if rec:
            self._auto_after_record(rec)
            if self.auto.state == AutoState.MISSING:
                self._auto_stop("漏装：未取得下一张有效照片，流程停止并报警")
            return
        if self.auto_worker and self.auto_worker.isRunning():
            return  # 上一张还没处理完，跳过本次节拍
        self.auto_worker = AutoShotWorker(self.backend, self.pipeline, self)
        self.auto_worker.shot_ready.connect(self._auto_on_shot)
        self.auto_worker.failed.connect(self._auto_on_shot_error)
        self.auto_worker.start()

    @Slot(object, object)
    def _auto_on_shot(self, frame, res):
        self.last_frame = frame
        qimg = bgr_to_qimage(res.overlay)
        if qimg and not self.preview.size().isEmpty():
            self.preview.setPixmap(QPixmap.fromImage(
                qimg.scaled(self.preview.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)))

        # A 遮挡帧的环状态必然 UNCERTAIN（工件A盖住轴承），不能当无效照片拒收，
        # 否则「检测到工件A → 安装结束」永远无法触发
        if (self.cfg.auto_flow.require_confident and res.ring_state == "UNCERTAIN"
                and not res.a_present):
            rec = self.auto.on_invalid(
                "检测失败：环状态 UNCERTAIN（置信不足）", event=AutoEvent.DETECT_FAIL)
        else:
            rec = self.auto.on_valid_photo(
                res.ring_state, res.shaft_present, res.a_present)
        self._auto_after_record(rec)

        if self.auto.state == AutoState.MISSING:
            self._auto_stop("漏装报警：请检查轴承后点「复位」")
            return
        if self.cfg.auto_flow.save_captures and rec is not None and rec.event in (
                AutoEvent.FLOW_START, AutoEvent.BEARING_OK,
                AutoEvent.CYCLE_OK, AutoEvent.MISSING_BEARING):
            folder = os.path.join(HERE, self.cfg.paths.captures, "auto")
            os.makedirs(folder, exist_ok=True)
            name = (f"{datetime.now():%Y%m%d_%H%M%S}_{rec.photo_no:03d}_"
                    f"{rec.event.value}.jpg")
            imwrite_unicode(os.path.join(folder, name), frame)

    @Slot(str, str)
    def _auto_on_shot_error(self, stage, reason):
        event = (AutoEvent.CAPTURE_FAIL if stage == "拍照失败"
                 else AutoEvent.DETECT_FAIL)
        rec = self.auto.on_invalid(f"{stage}：{reason}", event=event)
        self._auto_after_record(rec)
        if self.auto.state == AutoState.MISSING:
            self._auto_stop(f"{stage}过多：{reason}")

    _AUTO_NODE_FAULT_EVENTS = ("MISSING_BEARING", "PHOTO_TIMEOUT")

    def _auto_node(self, r) -> str:
        """自动拍照记录 → 记录表的业务流程节点（与流程图口径一致）。"""
        if r.alarm or r.event in self._AUTO_NODE_FAULT_EVENTS:
            return f"报警：{r.display}"
        if r.event == "FLOW_START":
            return "无轴承（检测到轴）"
        if r.event == "BEARING_OK":
            return "放置轴承（检测到轴承）"
        if r.event == "CYCLE_OK":
            return "安装结束/待机（检测到工件A）"
        if r.event in ("CAPTURE_FAIL", "DETECT_FAIL"):
            return "拍照/检测异常（重试）"
        if r.state == "IDLE":
            if r.a_present:
                return "安装结束/待机（工件A在位，重复检测）"
            return "安装结束/待机（等待检测到轴）"
        if r.state == "WAIT_BEARING":
            return "无轴承（等待放置轴承）"
        if r.state == "BEARING_PLACED":
            return "放置轴承（等待工件A）"
        if r.state == "MISSING":
            return "报警锁存"
        return r.display

    def _auto_after_record(self, rec):
        if rec is None:
            return
        self._auto_render_labels(self.auto.state)
        self.auto_detail_label.setText(
            f"[{rec.timestamp}] #{rec.photo_no} 节点：{self._auto_node(rec)}\n"
            f"{rec.message}\n"
            f"状态 {self.auto.display} ｜ 有效照片 {self.auto.valid_count} ｜ "
            f"拍照序号 {self.auto.photo_no} ｜ 轴承连拍 {self.auto.bearing_streak}"
            + (f"\n{self.last_result.notes}" if self.last_result else ""))
        self._auto_refresh_table()
        self._log(f"[自动拍照] {rec.state}/{rec.event} → {rec.display}：{rec.message}")
        if rec.alarm:
            self._auto_start_alarm()
        elif self.auto.state != AutoState.MISSING:
            color = {"IDLE": "#5F5E5A", "WAIT_BEARING": "#185FA5",
                     "BEARING_PLACED": "#0F6E56"}.get(self.auto.state.name, "#185FA5")
            self._set_status(rec.display, color)

    def _auto_refresh_table(self):
        """把自动拍照记录渲染进检测页的自动拍照记录表。"""
        recs = self.auto.records
        self.auto_table.setRowCount(len(recs))
        for row, r in enumerate(recs):
            cells = [str(r.index),
                     f"#{r.photo_no}" if r.photo_no else "-",
                     self._auto_node(r), r.display, r.message,
                     "是" if r.alarm else ""]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 2 and (r.alarm or r.event in self._AUTO_NODE_FAULT_EVENTS):
                    item.setForeground(Qt.darkRed)
                if col == 3:
                    if r.event == AutoEvent.CYCLE_OK.value:
                        item.setForeground(Qt.darkGreen)
                    elif r.state == AutoState.MISSING.value:
                        item.setForeground(Qt.darkRed)
                if col == 5 and r.alarm:
                    item.setForeground(Qt.darkRed)
                    item.setTextAlignment(Qt.AlignCenter)
                self.auto_table.setItem(row, col, item)
        if recs:
            self.auto_table.selectRow(len(recs) - 1)

    _AUTO_COLORS = {
        AutoState.IDLE: ("idle", "#5F5E5A", "#F1EFE8"),
        AutoState.WAIT_BEARING: ("no_bearing", "#185FA5", "#E6F1FB"),
        AutoState.BEARING_PLACED: ("placed", "#0F6E56", "#E1F5EE"),
        AutoState.MISSING: ("missing", "#A32D2D", "#FCEBEB"),
    }

    def _auto_render_labels(self, state):
        active_key = self._AUTO_COLORS[state][0]
        for key, lab in self.auto_labels.items():
            if key == active_key:
                fg, bg = self._AUTO_COLORS[state][1], self._AUTO_COLORS[state][2]
                lab.setStyleSheet(self._STYLE_ON.format(fg=fg, bg=bg))
                lab.setText(self.auto.display if key == active_key else lab.text())
            else:
                lab.setStyleSheet(self._STYLE_OFF)

    def _auto_start_alarm(self):
        self.auto_alarm_timer.start(600)
        if not self.auto_mute.isChecked():
            QApplication.beep()

    def _auto_blink_alarm(self):
        self.auto_alarm_on = not self.auto_alarm_on
        # 检测页无节点横幅：报警视觉提示改为状态行闪红，声音提示保持
        bg = "#F7C1C1" if self.auto_alarm_on else "#FCEBEB"
        self.auto_detail_label.setStyleSheet(
            f"color:#A32D2D;background:{bg};")
        if self.auto_alarm_on and not self.auto_mute.isChecked():
            QApplication.beep()

    def _auto_stop_alarm(self):
        self.auto_alarm_timer.stop()
        self.auto_detail_label.setStyleSheet("")
        for lab in self.auto_labels.values():
            lab.setStyleSheet(self._STYLE_OFF)

    def _auto_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出自动拍照记录", "auto_flow.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["序号", "照片序号", "流程节点", "状态", "事件", "显示",
                            "说明", "报警", "环状态", "主轴", "A到位", "时间"])
                for r in self.auto.records:
                    w.writerow([r.index, r.photo_no, self._auto_node(r), r.state,
                                r.event, r.display, r.message,
                                "是" if r.alarm else "", r.ring_state,
                                "是" if r.shaft_present else "",
                                "是" if r.a_present else "", r.timestamp])
            self._log(f"自动拍照记录已导出：{path}")
            QMessageBox.information(self, "完成", f"记录已导出：\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "失败", f"导出失败：{exc}")

    # ---------------------------------------------------------- 流程演示逻辑
    def _load_flow_ui_from_config(self):
        f = self.cfg.demo_flow
        self.flow_src.setEditText(f.source_dir)
        self.flow_dst.setEditText(f.target_dir)
        self.flow_interval.setValue(int(f.interval_ms))
        self.flow_loop.setChecked(bool(f.loop))
        self.flow_clear.setChecked(bool(f.clear_target))

    def _flow_sync_ui_to_config(self):
        f = self.cfg.demo_flow
        f.source_dir = self.flow_src.currentText().strip()
        f.target_dir = self.flow_dst.currentText().strip()
        f.interval_ms = self.flow_interval.value()
        f.loop = self.flow_loop.isChecked()
        f.clear_target = self.flow_clear.isChecked()
        self.flow.cfg = f

    def _flow_browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "选择示例照片来源目录", self.flow_src.currentText())
        if path:
            self.flow_src.setEditText(path)

    def _flow_browse_target(self):
        path = QFileDialog.getExistingDirectory(self, "选择目标文件夹（照片将按顺序放入）", self.flow_dst.currentText())
        if path:
            self.flow_dst.setEditText(path)

    def _flow_prepare(self):
        self._flow_sync_ui_to_config()
        ok, msg = self.flow.prepare(self.cfg.demo_flow.source_dir, self.cfg.demo_flow.target_dir)
        self._log(f"流程演示准备：{msg}")
        self._flow_refresh_sequence()
        self._flow_render_current()
        self._flow_refresh_table()
        self._flow_refresh_progress()
        self.flow_detail.setText(msg)
        if not ok:
            QMessageBox.information(self, "无法准备", msg)

    _FLOW_NODE_COLORS = {
        "idle": ("#0B7285", "#E3FAFC"),      # 安装结束/待机：青蓝（原灰色与未激活样式几乎一致，看不出高亮）
        "no_bearing": ("#185FA5", "#E6F1FB"),
        "placed": ("#0F6E56", "#E1F5EE"),
        "missing": ("#A32D2D", "#FCEBEB"),
    }
    _STYLE_PULSE = ("color:{fg};background:#FFFFFF;border:3px solid {fg};"
                    "border-radius:8px;padding:6px;")

    def _apply_banner_styles(self):
        """按当前激活节点与脉冲相位刷新横幅样式。"""
        for key, lab in self.flow_node_labels.items():
            if key == self._banner_active:
                fg, bg = self._FLOW_NODE_COLORS[key]
                if self._banner_pulse:
                    lab.setStyleSheet(self._STYLE_ON.format(fg=fg, bg=bg))
                else:
                    lab.setStyleSheet(self._STYLE_PULSE.format(fg=fg))
            else:
                lab.setStyleSheet(self._STYLE_OFF)

    def _banner_pulse_tick(self):
        self._banner_pulse = not self._banner_pulse
        self._apply_banner_styles()

    def _flow_render_banner(self, rec=None):
        active = self._flow_banner_key(rec)
        self._banner_active = active
        self._banner_pulse = True
        self._apply_banner_styles()
        if active:
            self._banner_timer.start()
        else:
            self._banner_timer.stop()

    def _flow_banner_key(self, rec):
        """把一条演示记录映射到横幅节点（安装结束/待机合并为一个节点）。"""
        if rec is None:
            return "idle"
        if rec.is_alarm or rec.state_after == "FAULT" or \
                getattr(rec, "node", "").startswith(("报警", "异常")):
            return "missing"
        if rec.state_after == "RING_READY" or rec.event == "CYCLE_START":
            return "placed"
        if rec.state_after == "IDLE" and not rec.a_present \
                and rec.ring_state == "ABSENT":
            return "no_bearing"   # 检测到轴 → 无轴承
        return "idle"             # 安装结束/待机（含检测到工件A之后）

    def _flow_reset(self):
        self._stop_flow_auto()
        self.flow.reset()
        self._flow_refresh_table()
        self._flow_refresh_progress()
        self._flow_refresh_sequence()
        self._flow_render_current()
        self.flow_badge.hide()
        self._flow_render_banner()
        self.flow_detail.setText("已重置，可从头逐步处理")
        self._log("流程演示已重置")

    def _flow_refresh_sequence(self):
        self.flow_seq_list.clear()
        total = self.flow.total()
        for i, path in enumerate(self.flow.target_images):
            name = os.path.basename(path)
            item = QListWidgetItem(f"{i + 1:02d}  ·  {name}")
            if i < len(self.flow.records):
                r = self.flow.records[i]
                mark = "…" if r.auto_verdict == "未判定" else \
                    ("✓" if r.auto_verdict == PASS else "✗")
                color = {"✓": "#0F6E56", "✗": "#A32D2D", "…": "#888888"}[mark]
                item.setText(f"{i + 1:02d}  {mark}  {name}")
                item.setForeground(QBrush(QColor(color)))
            self.flow_seq_list.addItem(item)
        if total:
            self.flow_seq_list.setCurrentRow(max(0, self.flow.cursor))

    def _flow_refresh_progress(self):
        done, total, pct = self.flow.progress()
        self.flow_progress.setValue(int(pct))
        p = sum(1 for r in self.flow.records if r.auto_verdict == PASS)
        f_ = sum(1 for r in self.flow.records if r.auto_verdict == FAIL)
        self.flow_progress_label.setText(
            f"{done} / {total} 张    ✓ 通过 {p}    ✗ 不通过 {f_}")

    def _flow_refresh_table(self):
        recs = self.flow.records
        self.flow_table.setRowCount(len(recs))
        for row, r in enumerate(recs):
            change = f"{r.state_before}→{r.state_after}" if r.state_before else "-"
            if r.auto_verdict == PASS:
                icon, verdict, fg_hex, bg_hex = "✓", PASS, "#0F6E56", "#E1F5EE"
            elif r.auto_verdict == FAIL:
                icon, verdict, fg_hex, bg_hex = "✗", FAIL, "#A32D2D", "#FCEBEB"
            else:
                icon, verdict, fg_hex, bg_hex = "…", "未判定", "#666666", "#FFFFFF"
            alarm = getattr(r, "is_alarm", False)
            cells = [str(r.index), r.filename, getattr(r, "node", ""),
                     f"{icon} {verdict}", f"{change} / {r.event}"]
            bg = QColor(bg_hex)
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(bg)
                if col == 3:                       # 自动判定列：按结果着色加粗
                    item.setForeground(QBrush(QColor(fg_hex)))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                elif col == 2:                     # 流程节点列：报警红、正常青
                    item.setForeground(QBrush(QColor("#A32D2D" if alarm else "#185FA5")))
                elif col == 1 and alarm:           # 漏装行的文件名也标红
                    item.setForeground(QBrush(QColor("#A32D2D")))
                self.flow_table.setItem(row, col, item)
        if recs:
            self.flow_table.selectRow(len(recs) - 1)

    def _flow_render_current(self):
        img = self.flow.current_image()
        if img is None:
            self.flow_preview.setText("当前无照片（先准备，或用上一张/下一张）")
            return
        # 若有判定结果，叠加管线标注；再由 GUI 画状态彩框与判定印章
        res = self.flow.last_result
        rec = self.flow.records[-1] if self.flow.records else None
        show = res.overlay if (res is not None and self.flow.records) else img
        qimg = bgr_to_qimage(show)
        if qimg and not self.flow_preview.size().isEmpty():
            pm = QPixmap.fromImage(qimg.scaled(
                self.flow_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._paint_demo_annotation(pm, res, rec, img)
            self.flow_preview.setPixmap(pm)

    @staticmethod
    def _detect_hough_circle(img_raw, hint_norm=None):
        """霍夫圆检测（仅演示展示用，不参与判定）。

        在原图上跑 cv2.HoughCircles（先缩到最长边 800 提速），
        返回原图坐标 (x, y, r)；hint_norm 为归一化提示点（环 ROI 中心），
        有提示时取离提示最近的圆，否则取最强圆。找不到返回 None。
        """
        if img_raw is None:
            return None
        h0, w0 = img_raw.shape[:2]
        scale = min(1.0, 800.0 / max(h0, w0))
        small = (cv2.resize(img_raw, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA) if scale < 1 else img_raw)
        gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
        sh, sw = gray.shape[:2]
        m = min(sh, sw)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(30.0, m / 2.0), param1=120, param2=35,
                                   minRadius=int(m * 0.06), maxRadius=int(m * 0.45))
        if circles is None:
            return None
        cands = [(float(x), float(y), float(r)) for x, y, r in circles[0]]
        if hint_norm:
            hx, hy = hint_norm[0] * sw, hint_norm[1] * sh
            cands.sort(key=lambda c: (c[0] - hx) ** 2 + (c[1] - hy) ** 2)
        x, y, r = cands[0]
        return (x / scale, y / scale, r / scale)

    def _paint_demo_annotation(self, pm, res, rec, img_raw=None):
        """在演示预览图上画判定条件彩框（环/A）与右上角 ✓/✗ 判定印章。

        纯展示层：环框颜色=环状态（绿=已放置/蓝=无环/琥珀=不确定），
        A框颜色=A 是否到位；印章按自动判定结果着色。不影响任何判定逻辑。
        """
        if pm.isNull():
            return
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = pm.width(), pm.height()

        def draw_state_roi(roi_norm, color, label):
            """画归一化 ROI 粗框 + 左上角标签牌；覆盖几乎全图的 ROI 跳过（无信息量）。"""
            if not roi_norm or len(roi_norm) < 4:
                return
            if roi_norm[2] * roi_norm[3] >= 0.95:
                return
            x0, y0 = roi_norm[0] * w, roi_norm[1] * h
            x1, y1 = (roi_norm[0] + roi_norm[2]) * w, (roi_norm[1] + roi_norm[3]) * h
            qc = QColor(color)
            p.setPen(QPen(qc, 3))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            # 标签牌：色底白字
            font = QFont("Microsoft YaHei", max(9, int(min(w, h) / 42)), QFont.Bold)
            p.setFont(font)
            fm = QFontMetrics(font)
            tw, th = fm.horizontalAdvance(label), fm.height()
            p.setPen(Qt.NoPen)
            p.setBrush(qc)
            p.drawRoundedRect(QRectF(x0, max(0, y0 - th - 8), tw + 10, th + 6), 4, 4)
            p.setPen(QColor("#FFFFFF"))
            p.drawText(QRectF(x0, max(0, y0 - th - 8), tw + 10, th + 6),
                       Qt.AlignCenter, label)

        if res is not None:
            model = self.flow.model
            ring_color = {"PLACED": "#21A366", "ABSENT": "#2F7BD9"}.get(
                getattr(res, "ring_state", ""), "#E6A817")
            ring_score = (res.ring_present_score if getattr(res, "ring_state", "") == "PLACED"
                          else res.ring_absent_score)
            ring_label = {"PLACED": "环：已放置", "ABSENT": "环：无环"}.get(
                getattr(res, "ring_state", ""), "环：不确定")
            if model is not None and getattr(model, "ring_roi", None):
                draw_state_roi(model.ring_roi, ring_color,
                               f"{ring_label} {ring_score:.2f}")
            a_color = "#E04B4B" if getattr(res, "a_present", False) else "#9AA0A6"
            a_label = (f"A：到位 {getattr(res, 'a_score', 0.0):.2f}"
                       if getattr(res, "a_present", False) else "A：未到")
            if model is not None and getattr(model, "a_roi", None):
                draw_state_roi(model.a_roi, a_color, a_label)

        # 霍夫圆环叠加（仅展示）：检出圆画紫色虚线圆 + 中心十字 + 文字标注
        if img_raw is not None:
            hint = None
            if res is not None and self.flow.model is not None:
                roi = getattr(self.flow.model, "ring_roi", None)
                if roi and len(roi) >= 4 and roi[2] * roi[3] < 0.95:
                    hint = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0)
            circle = self._detect_hough_circle(img_raw, hint)
            if circle is not None:
                hx, hy, hr = circle
                s = w / float(img_raw.shape[1])
                cx, cy, rr = hx * s, hy * s, hr * s
                hc = QColor("#9C27B0")            # 紫：与状态彩框（绿/蓝/琥珀/红）区分
                p.setPen(QPen(hc, 3, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx, cy), rr, rr)
                # 中心十字
                p.setPen(QPen(hc, 2))
                c = max(5, int(rr * 0.12))
                p.drawLine(QPointF(cx - c, cy), QPointF(cx + c, cy))
                p.drawLine(QPointF(cx, cy - c), QPointF(cx, cy + c))
                # 文字标注：圆心坐标与半径
                font = QFont("Microsoft YaHei", max(9, int(min(w, h) / 42)), QFont.Bold)
                p.setFont(font)
                fm = QFontMetrics(font)
                label = (f"霍夫圆 ({int(hx)},{int(hy)}) r={int(hr)}")
                tw, th = fm.horizontalAdvance(label), fm.height()
                lx = min(max(2, cx - rr), w - tw - 12)
                ly = max(2, cy - rr - th - 8)
                p.setPen(Qt.NoPen)
                p.setBrush(hc)
                p.drawRoundedRect(QRectF(lx, ly, tw + 10, th + 6), 4, 4)
                p.setPen(QColor("#FFFFFF"))
                p.drawText(QRectF(lx, ly, tw + 10, th + 6), Qt.AlignCenter, label)

        # 右上角判定印章
        if rec is not None:
            if rec.auto_verdict == PASS:
                text, color = "✓ 通过", "#21A366"
            elif rec.auto_verdict == FAIL:
                text, color = "✗ 漏装", "#E04B4B"
            else:
                text, color = "… 未判定", "#8A8A8A"
            font = QFont("Microsoft YaHei", max(12, int(min(w, h) / 20)), QFont.Bold)
            p.setFont(font)
            fm = QFontMetrics(font)
            tw, th = fm.horizontalAdvance(text), fm.height()
            pad = max(8, int(min(w, h) / 60))
            rect = QRectF(w - tw - 2 * pad - 14, 14, tw + 2 * pad, th + 2 * pad)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 215))
            p.drawRoundedRect(rect, 10, 10)
            p.setPen(QPen(QColor(color), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect, 10, 10)
            p.setPen(QColor(color))
            p.drawText(rect, Qt.AlignCenter, text)
        p.end()

    def _flow_set_badge(self, rec):
        """预览图左上角的判定徽章（✓ 通过 / ⚠ 漏装 / … 未判定）。"""
        if rec is None:
            self.flow_badge.hide()
            return
        if rec.auto_verdict == PASS:
            text, fg, bg = "✓ 通过", "#0F6E56", "#E1F5EE"
        elif rec.auto_verdict == FAIL:
            text, fg, bg = "⚠ 漏装", "#A32D2D", "#FCEBEB"
        else:
            text, fg, bg = "… 未判定", "#555555", "#F0F0F0"
        self.flow_badge.setText(text)
        self.flow_badge.setStyleSheet(
            f"color:{fg};background:{bg};border:2px solid {fg};border-radius:12px;"
            "padding:4px 14px;font-size:14px;font-weight:bold;")
        self.flow_badge.adjustSize()
        self.flow_badge.move(12, 12)
        self.flow_badge.raise_()
        self.flow_badge.show()

    def _flow_show_record(self, rec):
        if rec is None:
            return
        self._flow_render_banner(rec)
        self._flow_set_badge(rec)

        def span(text, color, bold=True):
            weight = "bold" if bold else "normal"
            return f'<span style="color:{color};font-weight:{weight}">{text}</span>'

        alarm = getattr(rec, "is_alarm", False)
        node = getattr(rec, "node", "")
        ring_color = {"PLACED": "#0F6E56", "ABSENT": "#185FA5"}.get(
            rec.ring_state, "#B8860B")                     # UNCERTAIN → 琥珀色
        if rec.event == "CYCLE_OK":
            ev_color = "#0F6E56"
        elif alarm or rec.state_after == "FAULT":
            ev_color = "#A32D2D"
        elif rec.event == "CYCLE_START":
            ev_color = "#185FA5"
        else:
            ev_color = "#555555"
        verdict_color = ("#0F6E56" if rec.auto_verdict == PASS
                         else "#A32D2D" if rec.auto_verdict == FAIL else "#666666")
        self.flow_detail.setText(
            f"{span(f'[{rec.index}/{self.flow.total()}] {rec.filename}', '#222')}\n"
            f"流程节点：{span(node, '#A32D2D' if alarm else '#185FA5')}\n"
            f"环={span(rec.ring_state, ring_color)}"
            f"  有环NCC={rec.ring_present_score:.2f} 无环NCC={rec.ring_absent_score:.2f}"
            f"  A={span('到位' if rec.a_present else '未到', '#A32D2D' if rec.a_present else '#555', False)}"
            f"({rec.a_score:.2f})\n"
            f"状态 {rec.state_before}→{rec.state_after}"
            f"  事件 {span(rec.event, ev_color)}\n"
            f"自动判定：{span(rec.auto_verdict, verdict_color)}\n"
            f"{rec.message}"
        )

    def _flow_next(self):
        if self.flow.total() == 0:
            self.flow_detail.setText(
                "还没有照片：请先点「准备照片」。\n来源目录读取不到图片时，"
                "会自动改用目标文件夹里已有的照片。")
            return
        rec = self.flow.step_next()
        if rec is None:
            self._stop_flow_auto()
            self.flow_detail.setText("已处理完最后一张。可重置重跑，或勾选循环。")
            return
        self._flow_after_step(rec)

    def _flow_after_step(self, rec):
        self._flow_refresh_sequence()
        self._flow_render_current()
        self._flow_show_record(rec)
        self._flow_render_banner(rec)
        self._flow_refresh_table()
        self._flow_refresh_progress()
        self._log(f"流程演示 [{rec.index}] {rec.filename} 节点={getattr(rec, 'node', '')} "
                  f"自动={rec.auto_verdict} 状态={rec.state_before}→{rec.state_after}")

    def _flow_prev(self):
        """回看上一张：仅移动光标查看照片与已有记录，不回退状态机。"""
        if self.flow.cursor > 0:
            self.flow.cursor -= 1
            self.flow_seq_list.setCurrentRow(self.flow.cursor)
            self._flow_render_current()
            if self.flow.cursor < len(self.flow.records):
                self._flow_show_record(self.flow.records[self.flow.cursor])
                self.flow_table.selectRow(self.flow.cursor)

    def _flow_goto(self, item):
        row = self.flow_seq_list.row(item)
        if row < 0:
            return
        # 只允许查看已处理过的张，避免跳过状态机
        if row < len(self.flow.records):
            self.flow.cursor = row
            self._flow_render_current()
            self._flow_show_record(self.flow.records[row])
            self.flow_table.selectRow(row)
        else:
            self.flow_detail.setText(f"第 {row + 1} 张尚未处理，请用“下一张（判定）”逐张推进。")

    def _flow_table_clicked(self, item):
        row = item.row()
        if row < len(self.flow.records) and row < self.flow.total():
            self.flow.cursor = row
            self.flow_seq_list.setCurrentRow(row)
            self._flow_render_current()
            self._flow_show_record(self.flow.records[row])

    def _flow_toggle_auto(self, checked=None):
        if self.flow_timer.isActive():
            self._stop_flow_auto()
        else:
            if self.flow.total() == 0:
                QMessageBox.information(self, "提示", "请先点“准备照片”")
                self.flow_auto_btn.setChecked(False)
                return
            self._flow_sync_ui_to_config()
            self.flow_timer.start(self.cfg.demo_flow.interval_ms)
            self.flow_auto_btn.setChecked(True)
            self.flow_auto_btn.setText("停止播放")
            self._log("流程演示自动播放已启动")

    def _flow_auto_step(self):
        if not self.flow.has_next():
            if self.cfg.demo_flow.loop and self.flow.total():
                self.flow.reset()
                self._flow_refresh_table()
            else:
                self._stop_flow_auto()
                return
        rec = self.flow.step_next()
        if rec:
            self._flow_after_step(rec)

    def _stop_flow_auto(self):
        if self.flow_timer.isActive():
            self.flow_timer.stop()
        self.flow_auto_btn.setChecked(False)
        self.flow_auto_btn.setText("自动播放")

    def _flow_save_config(self):
        self._flow_sync_ui_to_config()
        save_config(self.cfg, CONFIG_PATH)
        self._log(f"流程演示配置已保存：{CONFIG_PATH}")

    def _flow_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出流程报告", "flow_report.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        if self.flow.export_report(path):
            self._log(f"流程报告已导出：{path}")
            QMessageBox.information(self, "完成", f"报告已导出：\n{path}")
        else:
            QMessageBox.warning(self, "失败", "报告导出失败")

    def _reload_model(self):
        self.model = Model.load(os.path.join(HERE, self.cfg.paths.model))
        self.pipeline.set_model(self.model)
        self.flow.model = self.model
        self.flow.pipeline.set_model(self.model)
        self._log(f"模型重载：{'成功' if self.model else '未找到，使用纯几何法'}")

    def closeEvent(self, event):
        if self.auto_running:
            self._auto_stop("关闭程序，自动流程停止")
        self.auto_alarm_timer.stop()
        self._stop_grab()
        self._stop_flow_auto()
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
