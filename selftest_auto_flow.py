# -*- coding: utf-8 -*-
"""相机自动拍照流程无界面自测（不需要相机）。

用法：
    python selftest_auto_flow.py

覆盖（安装结束/待机合并版状态机）：
    1) 正常闭环：检测到轴→无轴承→放置轴承→检测到工件A→安装结束/待机
    2) 启动后拿不到下一张有效照片（超时）→ 漏装 + 报警
    3) 无轴承态下一张仍无轴承 → 漏装 + 报警
    4) 无轴承态出现工件A（A先于轴承）→ 漏装 + 报警
    5) 放置轴承后环在A到位前消失 → 漏装 + 报警
    6) 拍照失败后恢复 → 继续正常流程
    7) 连续失败超上限 → 漏装
    8) 主轴开关
    9) 安装结束后重复检测工件A → 自保持不报警
    10) 始终未检测到轴 → 持续安装结束/待机，不报警
    11) 待机中未拍裸轴直接检出轴承 → 顺序提示，按放置轴承继续
    12) 复位
    13) 真实三张现场图端到端判定（有轴无环 / 有轴承 / A遮挡）
"""
from __future__ import annotations

import os
import sys

from vision_app.auto_flow import AutoEvent, AutoFlowController, AutoState
from vision_app.config import AutoFlowConfig, load_config
from vision_app.detectors import VisionPipeline
from vision_app.imageio import imread_unicode
from vision_app.model import Model

HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        raise AssertionError(label)


def test_controller():
    print("=== 自动流程状态机（安装结束/待机合并版） ===")
    cfg = AutoFlowConfig()

    # 1) 正常闭环
    c = AutoFlowController(cfg)
    c.start()
    r1 = c.on_valid_photo("ABSENT", True, False, now=1.0)
    check(c.state == AutoState.WAIT_BEARING and r1.display == "无轴承",
          "检测到轴（有主轴、无轴承）→ 无轴承")
    r2 = c.on_valid_photo("PLACED", True, False, now=2.0)
    check(c.state == AutoState.BEARING_PLACED and r2.display == "放置轴承",
          "检测到轴承 → 放置轴承")
    r3 = c.on_valid_photo("UNCERTAIN", False, True, now=3.0)
    check(c.state == AutoState.IDLE and r3.display == "安装结束/待机"
          and r3.event == AutoEvent.CYCLE_OK.value,
          "检测到工件A → 安装结束/待机（本件完成）")
    check(not (r1.alarm or r2.alarm or r3.alarm), "正常闭环不报警")

    # 2) 启动后无下一张有效照片
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    r = c.check_timeout(now=1.0 + cfg.photo_timeout_ms / 1000.0 + 0.1)
    check(r is not None and c.state == AutoState.MISSING and r.alarm
          and r.event == AutoEvent.PHOTO_TIMEOUT.value,
          "启动后无下一张有效照片 → 漏装/报警")

    # 3) 无轴承态下一张仍无轴承
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    r = c.on_valid_photo("ABSENT", True, now=2.0)
    check(c.state == AutoState.MISSING and r.alarm
          and r.event == AutoEvent.MISSING_BEARING.value,
          "无轴承态下一张未检出轴承 → 漏装/报警")

    # 4) 无轴承态出现工件A（A先于轴承）
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    r = c.on_valid_photo("UNCERTAIN", False, True, now=2.0)
    check(c.state == AutoState.MISSING and r.alarm,
          "无轴承态工件A先于轴承出现 → 漏装/报警")

    # 5) 放置轴承后环在A到位前消失
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    c.on_valid_photo("PLACED", True, now=2.0)
    r = c.on_valid_photo("ABSENT", False, False, now=3.0)
    check(c.state == AutoState.MISSING and r.alarm,
          "环在A到位前消失 → 漏装/报警")

    # 6) 拍照失败后恢复
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    r = c.on_invalid("拍照失败：取帧超时", event=AutoEvent.CAPTURE_FAIL)
    check(r is not None and c.state == AutoState.WAIT_BEARING and not r.alarm,
          "单次拍照/检测失败仅重试，不误判漏装")
    c.on_valid_photo("PLACED", True, now=2.0)
    c.on_valid_photo("UNCERTAIN", False, True, now=3.0)
    check(c.state == AutoState.IDLE, "失败恢复后流程正常完成")

    # 7) 连续失败超上限
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    for _ in range(cfg.max_invalid_retries):
        r = c.on_invalid("检测失败：管线异常", event=AutoEvent.DETECT_FAIL)
    check(c.state == AutoState.MISSING and r.alarm,
          f"连续 {cfg.max_invalid_retries} 次无效照片 → 漏装/报警")

    # 8) 主轴开关
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", False, now=1.0)
    check(c.state == AutoState.IDLE, "未检出主轴时不启动流程（自保持）")
    cfg2 = AutoFlowConfig(require_shaft=False)
    c = AutoFlowController(cfg2)
    c.start()
    c.on_valid_photo("ABSENT", False, now=1.0)
    check(c.state == AutoState.WAIT_BEARING, "关闭主轴要求后可正常启动")

    # 9) 安装结束后重复检测工件A → 自保持
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    c.on_valid_photo("PLACED", True, now=2.0)
    c.on_valid_photo("UNCERTAIN", False, True, now=3.0)
    r = c.on_valid_photo("UNCERTAIN", False, True, now=4.0)
    check(c.state == AutoState.IDLE and not r.alarm
          and r.event == AutoEvent.NONE.value,
          "安装结束后重复检测工件A → 自保持，不报警")

    # 10) 始终未检测到轴 → 持续待机，不报警
    c = AutoFlowController(cfg)
    c.start()
    for t in (1.0, 2.0, 3.0):
        r = c.on_valid_photo("ABSENT", False, False, now=t)
    check(c.state == AutoState.IDLE and not r.alarm
          and r.event == AutoEvent.NONE.value,
          "始终未检测到轴 → 持续安装结束/待机，不报警、无超时")

    # 11) 待机中未拍裸轴直接检出轴承 → 顺序提示，按放置轴承继续
    c = AutoFlowController(cfg)
    c.start()
    r = c.on_valid_photo("PLACED", True, False, now=1.0)
    check(c.state == AutoState.BEARING_PLACED and not r.alarm
          and "顺序提示" in r.message,
          "待机中直接检出轴承 → 顺序提示，按放置轴承继续")
    r = c.on_valid_photo("UNCERTAIN", False, True, now=2.0)
    check(c.state == AutoState.IDLE and r.event == AutoEvent.CYCLE_OK.value,
          "顺序提示后检出工件A → 正常完成本件")

    # 12) 复位
    c = AutoFlowController(cfg)
    c.start()
    c.on_valid_photo("ABSENT", True, now=1.0)
    c.on_valid_photo("ABSENT", True, now=2.0)
    check(c.state == AutoState.MISSING, "前置：已进入漏装")
    c.reset()
    check(c.state == AutoState.IDLE and not c.records[-1].alarm,
          "复位回到安装结束/待机并清除报警锁存")


def test_real_images():
    print("=== 三张现场图端到端（主轴/轴承/A判定） ===")
    cfg = load_config(os.path.join(HERE, "config.json"))
    model = Model.load(os.path.join(HERE, cfg.paths.model))
    pipeline = VisionPipeline(cfg.detect, model)
    expected = [("1.jpg", "ABSENT", True, False), ("2.jpg", "PLACED", False, False),
                ("3.jpg", "UNCERTAIN", False, True)]
    for name, want_ring, want_shaft, want_a in expected:
        img = imread_unicode(os.path.join(HERE, "new_imgs", name))
        if img is None:
            print(f"  SKIP  {name} 不存在")
            continue
        res = pipeline.detect(img)
        print(f"  {name}: ring={res.ring_state} shaft={res.shaft_present} "
              f"a={res.a_present} ({res.a_score:.3f}) notes={res.notes}")
        check(res.ring_state == want_ring, f"{name} 环状态={want_ring}")
        check(res.shaft_present == want_shaft,
              f"{name} 主轴={'有' if want_shaft else '无'}")
        check(bool(res.a_present) == want_a, f"{name} 工件A={'到位' if want_a else '未到'}")


if __name__ == "__main__":
    test_controller()
    test_real_images()
    print("ALL PASS")
