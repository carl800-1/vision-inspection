# -*- coding: utf-8 -*-
"""离屏 GUI 冒烟：用三张现场图当文件相机，验证自动流程端到端。"""
from __future__ import annotations

import os
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ring_inspector_v2 import MainWindow

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    folder = os.path.join(HERE, "_auto_smoke_imgs")
    os.makedirs(folder, exist_ok=True)
    for src, dst in (("new_imgs/1.jpg", "01.jpg"),
                     ("new_imgs/2.jpg", "02.jpg"),
                     ("new_imgs/3.jpg", "03.jpg")):
        shutil.copy2(os.path.join(HERE, src), os.path.join(folder, dst))

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.cfg_source.setCurrentText("files")
    win.cfg_filedir.setEditText(folder)
    win.demo_btn.setChecked(False)
    win._apply_settings()   # 把设置页 UI 同步进 cfg，再用文件图源连接
    win._connect()
    win.auto_interval.setValue(200)
    win.auto_timeout.setValue(2000)
    win.tabs.setCurrentIndex(0)          # 检测页（自动拍照功能所在）
    win._auto_start()

    def done():
        print("tabs:", [win.tabs.tabText(i) for i in range(win.tabs.count())])
        print("state:", win.auto.state.value,
              "valid:", win.auto.valid_count,
              "events:", [r.event for r in win.auto.records], flush=True)
        for r in win.auto.records:
            print("   ", r.event, r.display, r.message, flush=True)
        # 文件图源循环播放，3.5s 截停时可能处于任一周节中间；
        # 只要完整走过至少一个周期即算通过
        events = [r.event for r in win.auto.records]
        assert "FLOW_START" in events and "CYCLE_OK" in events, events
        assert events[:4] == ["NONE", "FLOW_START", "BEARING_OK", "CYCLE_OK"], events
        win.close()
        app.quit()

    QTimer.singleShot(3500, done)
    app.exec()


if __name__ == "__main__":
    main()
