# -*- coding: utf-8 -*-
"""照片流程演示后端自测（无需 GUI / 相机）。

用法：
    python selftest_flow.py [来源照片目录]

默认来源：项目内 new_imgs（若无则用 dataset）。测试完整流程：
准备 → 逐张判定 → 人工判定 → 改规则重算 → 导出报告 → 重置重跑。
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vision_app.config import load_config
from vision_app.demo_flow import DemoFlow, PASS, FAIL
from vision_app.imageio import list_images
from vision_app.model import Model


def pick_source(explicit: str = "") -> str:
    if explicit:
        return explicit
    for cand in (os.path.join(HERE, "new_imgs"), os.path.join(HERE, "dataset")):
        if list_images(cand):
            return cand
    return os.path.join(HERE, "dataset")


def main():
    src = pick_source(sys.argv[1] if len(sys.argv) > 1 else "")
    cfg = load_config(os.path.join(HERE, "config.json"))
    tmp = tempfile.mkdtemp(prefix="flow_selftest_")
    dst = os.path.join(tmp, "inbox")

    f = cfg.demo_flow
    f.source_dir = src
    f.target_dir = dst
    f.judge_rule = "hybrid"
    f.copy_to_target = True
    f.clear_target = True

    model = Model.load(os.path.join(HERE, cfg.paths.model))
    flow = DemoFlow(f, cfg.detect, HERE, model)

    print(f"来源目录：{src}")
    print("=== 1. 准备（顺序放入目标文件夹）===")
    ok, msg = flow.prepare()
    print(f"  {ok} {msg}")
    print("  目标文件：" + ", ".join(os.path.basename(p) for p in flow.target_images))

    print("\n=== 2. 逐张判定 ===")
    fault_idx = []
    while flow.has_next():
        rec = flow.step_next()
        print(f"  [{rec.index}] {rec.filename}: 环={rec.ring_state} "
              f"A={'是' if rec.a_present else '否'} "
              f"状态 {rec.state_before}→{rec.state_after} 事件={rec.event} "
              f"自动={rec.auto_verdict} 最终={rec.final_verdict}")
        if rec.state_after == "FAULT":
            fault_idx.append(rec.index)

    # 漏装后自动复位（模拟实际操作）：FAULT 之后不允许停在锁存态
    for r in flow.records:
        if r.state_before == "FAULT":
            raise AssertionError(f"[{r.index}] 漏装后未自动复位，仍停在异常锁存")
    locked = [r.index for r in flow.records if "异常已锁存" in r.message]
    assert not locked, f"存在锁存记录：{locked}"
    print(f"  漏装出现在 {fault_idx}，均已自动复位，后续照片重新判定")

    print("\n=== 3. 人工判定（第1张通过，最后一张不通过）===")
    if flow.records:
        flow.judge(1, PASS)
        flow.judge(len(flow.records), FAIL)
        for r in flow.records:
            print(f"  [{r.index}] 人工={r.manual_verdict} 最终={r.final_verdict}")

    print("\n=== 4. 改规则为 auto 重算 ===")
    flow.set_rule("auto")
    print("  " + ", ".join(f"[{r.index}]={r.final_verdict}" for r in flow.records))

    print("\n=== 5. 导出报告 ===")
    report_csv = os.path.join(tmp, "report.csv")
    print(f"  导出={flow.export_report(report_csv)} 路径={report_csv}")
    print("  " + flow.report_lines()[-1])

    print("\n=== 6. 重置重跑（验证可重复）===")
    flow.reset()
    flow.set_rule("hybrid")
    print(f"  重置后记录数={len(flow.records)} 进度={flow.progress()[:2]}")
    while flow.has_next():
        rec = flow.step_next()
        print(f"  [{rec.index}] {rec.filename}: {rec.state_after} 自动={rec.auto_verdict}")
    print(f"  重跑完成，记录数={len(flow.records)}")

    # 场景7（可选）：含漏装的照片序列 → 验证报警后自动复位
    fault_src = r"D:\project\01.vision\v1.0.7\demo_flow_inbox"
    if os.path.isdir(fault_src) and list_images(fault_src):
        print("\n=== 7. 漏装后自动复位（模拟实际操作）===")
        f2 = cfg.demo_flow
        f2.source_dir = fault_src
        f2.target_dir = os.path.join(tmp, "inbox_fault")
        flow2 = DemoFlow(f2, cfg.detect, HERE, model)
        flow2.prepare()
        faults = []
        while flow2.has_next():
            rec = flow2.step_next()
            if rec.state_after == "FAULT":
                faults.append(rec.index)
        print(f"  漏装记录：{faults}")
        assert faults, "该场景应出现漏装"
        for r in flow2.records:
            assert r.state_before != "FAULT", \
                f"[{r.index}] 漏装后未自动复位，仍停在异常锁存"
            assert "异常已锁存" not in r.message, \
                f"[{r.index}] 出现锁存提示：{r.message}"
        nxt = [r for r in flow2.records if faults and r.index > faults[0]]
        assert nxt and nxt[0].state_before == "IDLE", "漏装后下一条应从待机重新判定"
        print("  漏装均已自动复位，后续照片从头判定 PASS")

    assert flow.total() >= 1, "未收集到照片"
    assert len(flow.records) == flow.total(), "记录数不等于照片数"
    assert all(os.path.isfile(p) for p in flow.target_images), "目标照片缺失"
    print("\nSELFTEST=OK")


if __name__ == "__main__":
    main()
