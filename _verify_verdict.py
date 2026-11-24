# -*- coding: utf-8 -*-
"""验证双标签判定区：PLACED 绿高亮 / ABSENT 红高亮闪烁。"""
import os
import sys
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.argv = ["verify"]
import ring_inspector as ri

OUT = r"C:\Users\Chenzhichao\WorkBuddy\2026-11-24-11-20-42"
app = QApplication(sys.argv)
win = ri.MainWindow()
win.show()


def style_of(label):
    ss = label.styleSheet()
    if "#E0E0E0" in ss or "#BBB" in ss:
        return "OFF(淡化)"
    if "#A32D2D" in ss:
        return "ON(红高亮)"
    if "#0F6E56" in ss:
        return "ON(绿高亮)"
    return "UNKNOWN:" + ss[:60]


def check(tag):
    p = style_of(win.placed_label)
    a = style_of(win.absent_label)
    print(f"[{tag}] 已放置={p}  未放置={a}")
    win.grab().save(os.path.join(OUT, f"verdict_{tag}.png"))


imgs = {
    "PLACED": cv2.imread(os.path.join(OUT, "demo_images", "with_ring_closeup.png")),
    "ABSENT": cv2.imread(os.path.join(OUT, "demo_images", "without_ring_closeup.png")),
}


def step1():
    res = ri.detect_ring(imgs["PLACED"])
    win._on_result(res, single=True)
    check("placed")


def step2():
    res = ri.detect_ring(imgs["ABSENT"])
    win._on_result(res, single=True)
    check("absent")


def step3():
    # 触发一次闪烁回调，确认红标签在两档红色间切换、绿标签保持淡化
    win._blink_alarm()
    check("absent_blink")
    print("计数:", win.count_label.text())
    app.quit()


QTimer.singleShot(200, step1)
QTimer.singleShot(600, step2)
QTimer.singleShot(1000, step3)
sys.exit(app.exec())
