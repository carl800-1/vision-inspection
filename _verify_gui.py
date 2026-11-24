# -*- coding: utf-8 -*-
"""GUI 端到端验证（offscreen）：演示模式连续采集、判定、截图。"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.argv = ["verify_gui"]

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from ring_inspector import MainWindow

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

app = QApplication(sys.argv)
win = MainWindow()
win.resize(1240, 820)
win.show()

win._start_grab()


def finish():
    win._stop_grab()
    shot = os.path.join(OUT_DIR, "gui_demo.png")
    win.grab().save(shot)
    print("=== 界面截图:", shot)
    print("=== 判定文字:", win.verdict_label.text())
    print("=== 计数:", win.count_label.text())
    print("=== 日志 ===")
    print(win.log.toPlainText())
    app.quit()


def snapshot_at(ms, name):
    """在指定时刻截一张图，用于观察 OK / NG 两种状态。"""
    def snap():
        shot = os.path.join(OUT_DIR, name)
        win.grab().save(shot)
        print(f"[{ms}ms] {name} -> {win.verdict_label.text()}")
    QTimer.singleShot(ms, snap)


# 每张图停留 5 帧 × 500ms = 2.5s，3 帧确认；第 4s 左右出第一个判定，第 7s 左右出第二个
snapshot_at(4500, "gui_state_ok.png")
snapshot_at(7000, "gui_state_ng.png")
QTimer.singleShot(8000, finish)
app.exec()
