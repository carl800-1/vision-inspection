# -*- coding: utf-8 -*-
"""环形轴承防错检测界面（PySide6）。

支持三种图源：
  1. 海康 MVS 工业相机（通过 hik_camera 封装，MV-CA050-10GC 等 GigE 型号）
  2. 演示模式（用两张特写图交替播放，无需相机即可验证界面和判定）
  3. 打开本地单张图片做一次判定

界面功能：连接/断开、参数配置、实时预览、单帧/连续采集、状态显示、
已放置/未放置/待复检三态、多帧确认、OK/NG 计数与报警。
"""
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTextEdit, QCheckBox, QMessageBox, QFrame, QStatusBar, QSizePolicy,
    QFileDialog,
)

from ring_detect_core import detect_ring, DEFAULT_CONFIG

def app_dir():
    """可写目录：源码运行时为脚本目录，打包后为 exe 所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(rel):
    """只读资源：兼容 PyInstaller onefile 的临时解压目录。"""
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", app_dir()), rel)
    return os.path.join(app_dir(), rel)


def prepare_mvs_dll_path():
    """把 MVS 运行时 DLL 目录加进搜索路径（裸名 WinDLL 加载所需）。

    实测完整 MVS 客户端把 MvCameraControl.dll 放在公共目录
    Common Files\\MVS\\Runtime\\Win64_x64，而非安装目录下的 Development\\Bin。
    """
    common = os.environ.get("CommonProgramFiles(x86)",
                            r"C:\Program Files (x86)\Common Files")
    sdk = os.environ.get("MVCAM_SDK_PATH", r"C:\Program Files (x86)\MVS")
    for d in (os.path.join(common, "MVS", "Runtime", "Win64_x64"),
              os.path.join(common, "MVS", "Runtime", "Win32_i86"),
              os.path.join(sdk, "Runtime", "x64"),
              os.path.join(sdk, "Runtime", "Win64_x64"),
              os.path.join(sdk, "Development", "Bin", "Win64"),
              os.path.join(sdk, "Development", "Bin", "win64"),
              os.path.join(sdk, "Development", "Samples", "Python", "MvImport")):
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass


HERE = app_dir()
CAPTURE_DIR = os.path.join(HERE, "captures")
DEMO_IMAGES = {
    "有环特写（已放置）": resource_path(os.path.join("demo_images", "with_ring_closeup.png")),
    "无环特写（未放置）": resource_path(os.path.join("demo_images", "without_ring_closeup.png")),
}


# --------------------------------------------------------------------------- #
# 相机后端抽象
# --------------------------------------------------------------------------- #
class CameraBackend:
    name = "base"

    def enumerate(self):
        return []

    def connect(self, device_id):
        raise NotImplementedError

    def disconnect(self):
        pass

    def configure(self, params):
        pass

    def grab(self):
        raise NotImplementedError


class DemoBackend(CameraBackend):
    """无相机演示：循环读取特写图，模拟真实取流。"""

    name = "DEMO"

    def __init__(self, source_key="交替循环", interval_ms=500, repeat=5):
        self.source_key = source_key
        self.interval_ms = interval_ms
        self.repeat = repeat      # 每张图连续出几帧，让多帧确认能生效
        self._idx = 0
        self._repeat = 0
        self._cache = {}
        for k, p in DEMO_IMAGES.items():
            img = cv2.imread(p)
            if img is not None:
                self._cache[k] = img

    def enumerate(self):
        return list(DEMO_IMAGES.keys()) + ["交替循环"]

    def connect(self, device_id):
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
        time.sleep(self.interval_ms / 1000.0)
        return img.copy()


class HikCameraBackend(CameraBackend):
    """海康 MVS 相机（hik_camera 封装）。须安装 MVS SDK。"""

    name = "HIK"

    def __init__(self):
        self.cam = None
        self._opened = False

    def enumerate(self):
        prepare_mvs_dll_path()
        from hik_camera import HikCamera
        # 用 _get_dev_info() 而非 get_all_ips()：后者打包后会用 sys.executable
        # 重新拉起 exe 导致递归；前者直接调 SDK 枚举，onefile 下安全。
        return list(HikCamera._get_dev_info().keys())

    def connect(self, device_id):
        prepare_mvs_dll_path()
        from hik_camera import HikCamera
        self.cam = HikCamera(device_id)
        # hik_camera 支持 with 语法，enter/exit 即 OpenDevice/CloseDevice
        self.cam.__enter__()
        self._opened = True
        return True

    def disconnect(self):
        if self._opened and self.cam is not None:
            try:
                self.cam.__exit__(None, None, None)
            except Exception:
                pass
        self._opened = False
        self.cam = None

    def configure(self, params):
        if not self._opened:
            return
        self.cam["ExposureAuto"] = "Off" if not params["exposure_auto"] else "Continuous"
        self.cam["ExposureTime"] = int(params["exposure_us"] * 1000)  # us -> ns
        self.cam["Gain"] = float(params["gain"])

    def grab(self):
        if not self._opened:
            return None
        rgb = self.cam.robust_get_frame()
        if rgb is None:
            return None
        if rgb.ndim == 2:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# 采集线程
# --------------------------------------------------------------------------- #
class GrabWorker(QThread):
    frame_ready = Signal(object)   # BGR ndarray
    result_ready = Signal(dict)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, backend, cfg_provider, detect_enabled=True, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.cfg_provider = cfg_provider
        self.detect_enabled = detect_enabled
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        self.status.emit("采集中")
        while not self._stop:
            try:
                frame = self.backend.grab()
            except Exception as exc:
                self.error.emit(f"取帧失败: {exc}")
                break
            if self._stop:
                break
            if frame is None:
                self.error.emit("取帧超时（连续 1 次）")
                continue
            self.frame_ready.emit(frame)
            if self.detect_enabled:
                try:
                    res = detect_ring(frame, self.cfg_provider())
                    self.result_ready.emit(res)
                except Exception as exc:
                    self.error.emit(f"判定异常: {exc}")
        self.status.emit("采集已停止")


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
def bgr_to_qimage(bgr):
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("环形轴承防错检测系统  ·  MV-CA050-10GC")
        self.resize(1240, 820)

        self.backend = None
        self.worker = None
        self.connected = False
        self.grabbing = False
        self.detect_on = True
        self.ok_count = 0
        self.ng_count = 0
        self.pending_verdict = None
        self.pending_count = 0
        self.last_frame = None
        self.current_verdict = None

        self._build_ui()
        self._apply_demo_mode()

        self.alarm_timer = QTimer(self)
        self.alarm_timer.timeout.connect(self._blink_alarm)
        self.alarm_on = False

    # ---------------- UI 构建 ---------------- #
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # 左侧：预览 + 结果
        left = QVBoxLayout()
        main.addLayout(left, 3)
        self._preview_box = QGroupBox("实时预览")

        self.preview = QLabel("未连接相机 / 演示模式")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#1a1a1a;color:#888;border:1px solid #444;")
        self._preview_box.setLayout(QVBoxLayout())
        self._preview_box.layout().addWidget(self.preview)

        result_box = QFrame()
        result_box.setFrameShape(QFrame.StyledPanel)
        rb = QHBoxLayout(result_box)
        verdict_box = QVBoxLayout()
        verdict_row = QHBoxLayout()
        vf = QFont("Microsoft YaHei", 22, QFont.Bold)
        self.placed_label = QLabel("已放置 ✓")
        self.placed_label.setFont(vf)
        self.placed_label.setAlignment(Qt.AlignCenter)
        self.placed_label.setMinimumHeight(58)
        self.absent_label = QLabel("未放置 ✗")
        self.absent_label.setFont(vf)
        self.absent_label.setAlignment(Qt.AlignCenter)
        self.absent_label.setMinimumHeight(58)
        verdict_row.addWidget(self.placed_label, 1)
        verdict_row.addWidget(self.absent_label, 1)
        verdict_box.addLayout(verdict_row)
        self.detail_label = QLabel("等待图像…")
        self.detail_label.setFont(QFont("Consolas", 10))
        self.detail_label.setWordWrap(True)
        verdict_box.addWidget(self.detail_label)
        rb.addLayout(verdict_box, 4)
        self._reset_verdict_style()
        left.addWidget(self._preview_box)
        left.addWidget(result_box)

        # 右侧：控制区
        right = QVBoxLayout()
        main.addLayout(right, 1)

        # 相机连接
        g1 = QGroupBox("相机连接")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)  # 枚举异常时可手动输入相机 IP
        self.source_combo.setPlaceholderText("选择或手动输入相机 IP")
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
        self.demo_btn.clicked.connect(self._apply_demo_mode)
        row2.addWidget(self.connect_btn)
        row2.addWidget(self.demo_btn)
        v1.addLayout(row2)
        self.status_label = QLabel("状态：未连接")
        self.status_label.setStyleSheet("color:#666;font-weight:bold;")
        v1.addWidget(self.status_label)
        right.addWidget(g1)

        # 采集控制
        g2 = QGroupBox("采集控制")
        v2 = QVBoxLayout(g2)
        row3 = QHBoxLayout()
        self.single_btn = QPushButton("单帧采集")
        self.single_btn.clicked.connect(self._single_capture)
        self.cont_btn = QPushButton("连续采集")
        self.cont_btn.setCheckable(True)
        self.cont_btn.clicked.connect(self._toggle_grab)
        row3.addWidget(self.single_btn)
        row3.addWidget(self.cont_btn)
        v2.addLayout(row3)
        row4 = QHBoxLayout()
        self.detect_chk = QCheckBox("实时判定")
        self.detect_chk.setChecked(True)
        row4.addWidget(self.detect_chk)
        self.open_img_btn = QPushButton("打开图片判定")
        self.open_img_btn.clicked.connect(self._open_image)
        row4.addWidget(self.open_img_btn)
        v2.addLayout(row4)
        right.addWidget(g2)

        # 相机参数
        g3 = QGroupBox("相机参数")
        form = QGridLayout(g3)
        form.addWidget(QLabel("曝光 (μs)"), 0, 0)
        self.exposure = QSpinBox()
        self.exposure.setRange(1, 1000000)
        self.exposure.setValue(20000)
        form.addWidget(self.exposure, 0, 1)
        form.addWidget(QLabel("增益 (dB)"), 1, 0)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 24)
        self.gain.setValue(0)
        form.addWidget(self.gain, 1, 1)
        self.exp_auto = QCheckBox("自动曝光（不建议，判定基准会漂移）")
        form.addWidget(self.exp_auto, 2, 0, 1, 2)
        self.apply_param_btn = QPushButton("应用参数")
        self.apply_param_btn.clicked.connect(self._apply_params)
        form.addWidget(self.apply_param_btn, 3, 0, 1, 2)
        right.addWidget(g3)

        # 判定参数
        g4 = QGroupBox("判定参数")
        form2 = QGridLayout(g4)
        form2.addWidget(QLabel("暗谷阈值"), 0, 0)
        self.th_spin = QSpinBox()
        self.th_spin.setRange(10, 150)
        self.th_spin.setValue(int(DEFAULT_CONFIG["valley_th"]))
        form2.addWidget(self.th_spin, 0, 1)
        form2.addWidget(QLabel("ROI 比例"), 1, 0)
        self.roi_spin = QDoubleSpinBox()
        self.roi_spin.setRange(0.3, 0.95)
        self.roi_spin.setSingleStep(0.05)
        self.roi_spin.setValue(DEFAULT_CONFIG["roi_ratio"])
        form2.addWidget(self.roi_spin, 1, 1)
        form2.addWidget(QLabel("确认帧数"), 2, 0)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 10)
        self.frames_spin.setValue(3)
        form2.addWidget(self.frames_spin, 2, 1)
        right.addWidget(g4)

        # 统计
        g5 = QGroupBox("统计与日志")
        v5 = QVBoxLayout(g5)
        self.count_label = QLabel("已放置 0   |   未放置 0")
        self.count_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        v5.addWidget(self.count_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        v5.addWidget(self.log)
        right.addWidget(g5)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪 · 演示模式可无相机运行")

    # ---------------- 配置 ---------------- #
    def _cfg(self):
        return {
            "valley_th": float(self.th_spin.value()),
            "roi_ratio": float(self.roi_spin.value()),
        }

    def _log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

    def _set_status(self, text, color="#666"):
        self.status_label.setText(f"状态：{text}")
        self.status_label.setStyleSheet(f"color:{color};font-weight:bold;")

    # ---------------- 连接管理 ---------------- #
    def _refresh_devices(self):
        if self.demo_btn.isChecked():
            self.source_combo.clear()
            self.source_combo.addItems(DemoBackend().enumerate())
            return
        try:
            devices = HikCameraBackend().enumerate()
            self.source_combo.clear()
            self.source_combo.addItems(devices if devices else ["未发现相机"])
            self._log(f"枚举设备：{len(devices)} 台")
        except Exception as exc:
            self._log(f"枚举失败：{exc}")
            QMessageBox.warning(self, "枚举失败", f"{exc}\n\n请确认已安装 MVS SDK。")

    def _apply_demo_mode(self, checked=None):
        if self.connected or self.grabbing:
            self._stop_grab()
            self._disconnect()
        self.demo_btn.setChecked(True)
        self.connect_btn.setText("连接")
        self.source_combo.clear()
        self.source_combo.addItems(DemoBackend().enumerate())
        self.source_combo.setCurrentText("交替循环")
        self._set_status("演示模式（未连接相机）", "#185FA5")
        self._log("已切换演示模式：使用本地特写图模拟取流")

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        if self.demo_btn.isChecked():
            self._log("演示模式下请使用「连续采集」开始预览")
            return
        device = self.source_combo.currentText()
        if not device or device == "未发现相机":
            QMessageBox.information(self, "提示", "请先枚举并选择相机")
            return
        try:
            self.backend = HikCameraBackend()
            self.backend.connect(device)
            self.connected = True
            self.connect_btn.setText("断开")
            self.backend.configure(self._camera_params())
            self._set_status(f"已连接 {device}", "#0F6E56")
            self._log(f"相机已连接：{device}")
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
        self._log("相机已断开")

    def _camera_params(self):
        return {
            "exposure_auto": self.exp_auto.isChecked(),
            "exposure_us": self.exposure.value(),
            "gain": self.gain.value(),
        }

    def _apply_params(self):
        if self.backend and self.connected:
            try:
                self.backend.configure(self._camera_params())
                self._log("相机参数已下发")
            except Exception as exc:
                self._log(f"参数下发失败：{exc}")
        else:
            self._log("未连接真实相机，参数仅保存在界面")

    # ---------------- 采集 ---------------- #
    def _ensure_backend(self):
        if self.demo_btn.isChecked():
            if not isinstance(self.backend, DemoBackend):
                self.backend = DemoBackend(self.source_combo.currentText() or "交替循环")
            return True
        if not self.connected:
            QMessageBox.information(self, "提示", "请先连接相机")
            return False
        return True

    def _start_grab(self):
        if not self._ensure_backend():
            return
        if isinstance(self.backend, DemoBackend):
            self.backend = DemoBackend(self.source_combo.currentText() or "交替循环")
        self.worker = GrabWorker(self.backend, self._cfg, self.detect_chk.isChecked())
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
        self._reset_verdict_style()

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
            self._log(f"单帧采集失败：{exc}")
            return
        if frame is None:
            self._log("单帧采集失败：取帧超时")
            return
        self._on_frame(frame)
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(os.path.join(CAPTURE_DIR, name), frame)
        res = detect_ring(frame, self._cfg())
        self._on_result(res, single=True)
        self._log(f"单帧已保存：{name}")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", "图片读取失败")
            return
        self._on_frame(img)
        res = detect_ring(img, self._cfg())
        self._on_result(res, single=True)
        self._log(f"本地图片判定：{os.path.basename(path)}")

    # ---------------- 结果与防错 ---------------- #
    @Slot(object)
    def _on_frame(self, frame):
        self.last_frame = frame
        qimg = bgr_to_qimage(frame)
        if qimg:
            scaled = qimg.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(QPixmap.fromImage(scaled))

    @Slot(dict)
    def _on_result(self, res, single=False):
        verdict = res["verdict"]
        # 连续采集时做多帧确认，单帧/打开图片直接出结果
        if not single and self.detect_chk.isChecked():
            need = self.frames_spin.value()
            if verdict == self.pending_verdict:
                self.pending_count += 1
            else:
                self.pending_verdict = verdict
                self.pending_count = 1
            if self.pending_count < need:
                self.detail_label.setText(
                    f"确认中 {self.pending_count}/{need}\n"
                    f"本帧：{verdict}  暗谷={res['valley']:.0f}")
                return
            self.pending_count = 0

        self.current_verdict = verdict
        self._show_verdict(verdict, res)
        if verdict == "PLACED":
            self.ok_count += 1
        elif verdict == "ABSENT":
            self.ng_count += 1
        self.count_label.setText(f"已放置 {self.ok_count}   |   未放置 {self.ng_count}")
        self._log(
            f"判定={verdict} 暗谷={res['valley']:.0f}@{res['valley_r']}px "
            f"亮峰={res['peak']:.0f} 落差={res['contrast']:.0f} 裕度={res['margin']:.2f}")

    # 判定标签样式：命中高亮、另一个淡化
    _STYLE_ON = "color:{fg};background:{bg};border:2px solid {fg};border-radius:8px;padding:6px;"
    _STYLE_OFF = "color:#BBB;background:#F5F5F5;border:2px solid #E0E0E0;border-radius:8px;padding:6px;"

    def _reset_verdict_style(self):
        self.placed_label.setStyleSheet(self._STYLE_OFF)
        self.absent_label.setStyleSheet(self._STYLE_OFF)

    def _show_verdict(self, verdict, res):
        # 两个常驻标签，命中的高亮、另一个淡化
        if verdict == "PLACED":
            self.placed_label.setStyleSheet(
                self._STYLE_ON.format(fg="#0F6E56", bg="#E1F5EE"))
            self.absent_label.setStyleSheet(self._STYLE_OFF)
            fg = "#0F6E56"
        elif verdict == "ABSENT":
            self.placed_label.setStyleSheet(self._STYLE_OFF)
            self.absent_label.setStyleSheet(
                self._STYLE_ON.format(fg="#A32D2D", bg="#FCEBEB"))
            fg = "#A32D2D"
        else:  # UNCERTAIN 待复检：两个都淡化，状态栏提示
            self._reset_verdict_style()
            fg = "#854F0B"

        self.detail_label.setText(
            f"暗谷 {res['valley']:.0f} @ r={res['valley_r']}px   阈值 {self.th_spin.value()}\n"
            f"亮峰 {res['peak']:.0f} @ r={res['peak_r']}px   落差 {res['contrast']:.0f}\n"
            f"置信裕度 {res['margin']:.2f}   中心圆 r={res['circle'][2]}px")
        # 判定叠加图
        qimg = bgr_to_qimage(res["overlay"])
        if qimg:
            scaled = qimg.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(QPixmap.fromImage(scaled))

        if verdict == "ABSENT":
            self._set_status("报警：未放置环形轴承", "#A32D2D")
            self.alarm_on = False
            self.alarm_timer.start(500)   # 红色闪烁
        else:
            self.alarm_timer.stop()
            self._set_status("正常" if verdict == "PLACED" else "待复检", fg)

    def _blink_alarm(self):
        self.alarm_on = not self.alarm_on
        bg = "#F7C1C1" if self.alarm_on else "#FCEBEB"
        self.placed_label.setStyleSheet(self._STYLE_OFF)
        self.absent_label.setStyleSheet(
            self._STYLE_ON.format(fg="#A32D2D", bg=bg))

    @Slot(str)
    def _on_error(self, msg):
        self._log(f"异常：{msg}")
        self._set_status("异常，请检查相机/网络", "#A32D2D")
        self.statusbar.showMessage(msg)

    def closeEvent(self, event):
        self._stop_grab()
        if self.backend:
            self.backend.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    icon_path = resource_path("app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    win = MainWindow()
    win.show()

    # 无相机端到端自检：python ring_inspector.py --selftest
    # 打包后可用 RingInspector.exe --selftest，结果写到 exe 同目录
    if "--selftest" in sys.argv:
        win._start_grab()

        def _finish_selftest():
            win._stop_grab()
            out = app_dir()
            win.grab().save(os.path.join(out, "selftest.png"))
            with open(os.path.join(out, "selftest.txt"), "w", encoding="utf-8") as f:
                f.write("verdict=%s\n" % win.current_verdict)
                f.write("counts=%s\n" % win.count_label.text())
                f.write("--- log tail ---\n%s\n" % win.log.toPlainText()[-800:])
            app.quit()

        QTimer.singleShot(4000, _finish_selftest)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
