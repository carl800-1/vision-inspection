# -*- coding: utf-8 -*-
"""用 state_1/2/3 三张图模拟完整检测周期。

模拟设置：
  - 使用 config.json 中按现场三图标定的真实参数
  - 环判定：几何暗谷阈值；A判定：偏心孔 + 横边 + 模板投票，并要求中心清空
  - 每张图连续送 3 帧，满足确认帧数后触发状态转移
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from vision_app.config import load_config
from vision_app.detectors import VisionPipeline
from vision_app.imageio import imread_unicode, imwrite_unicode, list_images
from vision_app.model import Model
from vision_app.state_machine import PokerStateMachine, StableSignal

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = load_config(os.path.join(HERE, "config.json"))
    d = cfg.detect
    # 直接用 config.json 的现场标定参数，不做任何模拟专用覆盖

    model = Model.load(os.path.join(HERE, cfg.paths.model))
    pipeline = VisionPipeline(d, model)
    sm = PokerStateMachine(d.ring_removal_is_fault)
    ring_s = StableSignal(d.ring_confirm_frames)
    a_s = StableSignal(d.a_confirm_frames)

    files = list_images(os.path.join(HERE, "sim_images"), recursive=False)
    if len(files) != 3:
        raise RuntimeError(f"sim_images 需要 3 张图片，实际 {len(files)} 张")

    lines = ["=== 1→2→3 模拟报告 ===",
             f"阈值 valley_th={d.valley_th}  ring_mode={d.ring_mode}  "
             f"a_mode={d.a_feature_mode}  require_center_clear={d.require_center_clear}",
             ""]
    annotated = []

    for idx, path in enumerate(files, 1):
        img = imread_unicode(path)
        if img is None:
            raise RuntimeError(f"读取失败：{path}")
        # 每张图重复确认帧数，模拟真实相机连续取流
        kept = None
        for _ in range(max(d.ring_confirm_frames, d.a_confirm_frames)):
            res = pipeline.detect(img)
            ring, _ = ring_s.feed(res.ring_state)
            a_present, _ = a_s.feed(res.a_present)
            state, event, msg = sm.feed(ring, a_present)
            if kept is None or event.value != "NONE":
                kept = (state, event, msg)
        state, event, msg = kept

        lines.append(
            f"[{idx}] {os.path.basename(path)}: ring={res.ring_state} "
            f"valley={res.geometry.get('valley'):.1f} "
            f"A_score={res.a_score:.3f} A={res.a_present} "
            f"→ state={state.value} event={event.value} msg={msg}"
        )
        ov = res.overlay.copy()
        cv2.putText(ov, f"STEP {idx}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 0), 2, cv2.LINE_AA)
        annotated.append(ov)
        imwrite_unicode(os.path.join(HERE, f"sim_state_{idx}_annotated.jpg"), ov)

    lines += ["", f"最终状态={sm.state.value}  OK={sm.ok_count}  异常={sm.fault_count}"]

    # 拼一张总览图
    target_w = 520
    panels = []
    for ov in annotated:
        h, w = ov.shape[:2]
        scale = target_w / float(w)
        panels.append(cv2.resize(ov, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA))
    composite = np.hstack(panels)
    imwrite_unicode(os.path.join(HERE, "simulation_1_2_3.png"), composite)

    report = os.path.join(HERE, "simulation_report.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n标注图：simulation_1_2_3.png\n报告：{report}")


if __name__ == "__main__":
    main()
