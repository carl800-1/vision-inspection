# -*- coding: utf-8 -*-
"""照片流程演示后端。

模拟"一组示例照片按顺序进入待处理文件夹 → 逐张查看 → 判定通过/不通过 →
记录结果与状态变化 → 重置后重跑"的完整流程。判定沿用现有的检测管线与
事件驱动状态机，因此演示行为与真实检测一致。

设计要点：
    - 来源目录：普通图片目录（按文件名自然排序）或数据集根目录
      （含 state1_no_ring / state2_with_ring / state3_workpiece_a，按 1→2→3 排序）。
    - 目标文件夹：把来源照片以 001_/002_... 序号前缀复制进去，保证顺序可见、可重复。
    - 每条记录含：序号、文件名、自动环状态/评分、A 状态、自动判定、人工判定、
      最终判定、状态机前后状态、事件、说明、时间。
    - 判断规则 judge_rule：
        manual  最终判定 = 人工判定；
        auto    最终判定 = 自动判定；
        hybrid  自动给建议，人工确认；最终取人工（人工未判时用自动建议）。
"""
from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import AppConfig, DemoFlowConfig
from .dataset import DatasetManager
from .detectors import VisionPipeline
from .imageio import imread_unicode, list_images
from .model import Model
from .state_machine import CycleState, Event, PokerStateMachine, StableSignal

PASS = "通过"
FAIL = "不通过"
PENDING = "未判定"

# 业务流程节点（正式界面只显示结果，演示时把节点轨迹完整亮出来便于核对）
FAULT_EVENTS = {
    "FAULT_A_FIRST": "报警：工件A先于轴承出现（漏放轴承）",
    "FAULT_RING_REMOVED": "报警：轴承在工件前被取走",
}


def _natural_key(name: str):
    """文件名自然排序：img2 排在 img10 前面。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


@dataclass
class FlowRecord:
    """单张照片的一条处理记录。"""
    index: int = 0
    filename: str = ""
    source: str = ""
    ring_state: str = ""
    ring_present_score: float = 0.0
    ring_absent_score: float = 0.0
    a_present: bool = False
    a_score: float = 0.0
    state_before: str = ""
    state_after: str = ""
    event: str = ""
    node: str = ""          # 业务流程节点（循环开始/放置轴承/放置工件/报警…）
    auto_verdict: str = ""
    manual_verdict: str = ""
    final_verdict: str = ""
    message: str = ""
    notes: str = ""
    timestamp: str = ""

    @property
    def is_alarm(self) -> bool:
        return self.node.startswith(("报警", "异常"))


class DemoFlow:
    """照片流程演示控制器。所有路径相对 base_dir 解析，兼容 exe 目录。"""

    def __init__(self, cfg: DemoFlowConfig, detect_cfg, base_dir: str,
                 model: Optional[Model] = None):
        self.cfg = cfg
        self.detect_cfg = detect_cfg
        self.base_dir = base_dir
        self.model = model
        self.pipeline = VisionPipeline(detect_cfg, model)
        self.sm = PokerStateMachine(detect_cfg.ring_removal_is_fault)
        self.ring_stable = StableSignal(detect_cfg.ring_confirm_frames)
        self.a_stable = StableSignal(detect_cfg.a_confirm_frames)

        self.source_images: List[str] = []   # 来源照片（绝对/原始路径）
        self.target_images: List[str] = []   # 目标文件夹里的照片（按顺序）
        self.records: List[FlowRecord] = []
        self.cursor: int = -1                # 当前处理到的下标，-1=未开始
        self.last_result = None
        self.last_frame = None
        self._last_event = Event.NONE
        self._last_message = ""

    # ------------------------------------------------------------ 路径解析
    def _resolve(self, path: str) -> str:
        if not path:
            return os.path.abspath(os.path.join(self.base_dir, "demo_flow"))
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(self.base_dir, path))

    def source_dir(self) -> str:
        return self._resolve(self.cfg.source_dir)

    def target_dir(self) -> str:
        return self._resolve(self.cfg.target_dir)

    # ------------------------------------------------------------ 准备阶段
    def collect_source(self) -> List[str]:
        """收集来源照片，数据集根目录按状态1→2→3排序，普通目录按文件名自然排序。

        容错：标准子目录存在但为空、或顶层没有图片时，回退扫描整个根目录，
        避免"建了空子目录/图片放错一层"导致演示完全无图可用。
        """
        root = self.source_dir()
        if not os.path.isdir(root):
            return []
        # 识别数据集根目录（含任一标准状态子目录）
        subdirs = [d for d in ("state1_no_ring", "state2_with_ring", "state3_workpiece_a")
                   if os.path.isdir(os.path.join(root, d))]
        if subdirs:
            scanned = DatasetManager(root).scan()
            ordered = scanned["absent"] + scanned["present"] + scanned["a"]
            if ordered:
                return ordered
            # 子目录全是空的：回退为普通目录扫描（根目录本身的图片仍可用）
        # 普通目录：优先只取顶层（避免把无关子文件夹的照片混进演示顺序）
        top = [p for p in list_images(root, recursive=False)
               if os.path.dirname(p) == root]
        if top:
            return sorted(top, key=lambda p: _natural_key(os.path.basename(p)))
        return sorted(list_images(root, recursive=True),
                      key=lambda p: _natural_key(os.path.basename(p)))

    def prepare(self, source_dir: Optional[str] = None,
                target_dir: Optional[str] = None) -> Tuple[bool, str]:
        """把来源照片按顺序复制进目标文件夹，并重置演示状态。"""
        if source_dir is not None:
            self.cfg.source_dir = source_dir
        if target_dir is not None:
            self.cfg.target_dir = target_dir
        src = self.collect_source()
        if not src:
            # 兜底：来源目录没有照片，但目标文件夹里已有照片时直接用它们，
            # 避免"照片放进了 inbox、来源目录留空"时演示无反应。
            tgt_files = list_images(self.target_dir(), recursive=False)
            if tgt_files:
                self.source_images = []
                self.target_images = sorted(
                    tgt_files, key=lambda p: _natural_key(os.path.basename(p)))
                self.reset()
                return True, (f"来源目录没有照片，已直接使用目标文件夹中现有的 "
                              f"{len(self.target_images)} 张照片（按文件名排序，未复制）")
            self.source_images = []
            self.target_images = []
            return False, f"来源目录没有图片：{self.source_dir()}（目标文件夹也没有照片）"
        self.source_images = list(src)

        dst_dir = self.target_dir()
        try:
            if self.cfg.clear_target and os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            os.makedirs(dst_dir, exist_ok=True)
            if self.cfg.copy_to_target:
                targets = []
                for i, p in enumerate(src, 1):
                    dst = os.path.join(dst_dir, f"{i:03d}_{os.path.basename(p)}")
                    shutil.copy2(p, dst)
                    targets.append(dst)
                self.target_images = targets
            else:
                self.target_images = list(src)
        except Exception as exc:
            return False, f"准备目标文件夹失败：{exc}"

        self.reset()
        ok = bool(self.target_images)
        return ok, (f"已放入 {len(self.target_images)} 张照片 → {dst_dir}" if ok
                    else "目标文件夹没有照片")

    # ------------------------------------------------------------ 流程控制
    def reset(self) -> None:
        """清空记录与状态机，照片序列保留，可从头重跑。"""
        self.records = []
        self.cursor = -1
        self.sm.reset()
        self.ring_stable.reset()
        self.a_stable.reset()
        self.last_result = None
        self.last_frame = None
        self._last_event = Event.NONE
        self._last_message = ""

    def total(self) -> int:
        return len(self.target_images)

    def progress(self) -> Tuple[int, int, float]:
        total = self.total()
        done = len(self.records)
        pct = (done / total * 100.0) if total else 0.0
        return done, total, pct

    def current_path(self) -> Optional[str]:
        if 0 <= self.cursor < self.total():
            return self.target_images[self.cursor]
        return None

    def current_image(self):
        path = self.current_path()
        return imread_unicode(path) if path else None

    def _infer_expected(self, path: str) -> str:
        """从路径/文件名推断期望状态，用于报告对比（非判定依据）。"""
        s = path.replace("\\", "/").lower()
        if any(k in s for k in ("state3", "workpiece_a", "3_workpiece", "_a.", "工件a")):
            return "A到位"
        if any(k in s for k in ("state2", "with_ring", "present", "2_with", "有环")):
            return "有环"
        if any(k in s for k in ("state1", "no_ring", "absent", "1_no", "无环")):
            return "无环"
        return ""

    def has_next(self) -> bool:
        return self.cursor + 1 < self.total()

    def step_next(self, repeat: int = 0) -> Optional[FlowRecord]:
        """处理下一张照片。repeat=每张照片连续送帧数（满足确认帧数）。"""
        if not self.has_next():
            return None
        self.cursor += 1
        path = self.target_images[self.cursor]
        img = imread_unicode(path)
        if img is None:
            rec = FlowRecord(index=self.cursor + 1, filename=os.path.basename(path),
                             source=path, message="图片读取失败",
                             node="异常：照片读取失败", timestamp=self._now())
            self.records.append(rec)
            return rec

        self.last_frame = img
        frames = max(1, repeat, self.detect_cfg.ring_confirm_frames,
                     self.detect_cfg.a_confirm_frames)
        kept = None
        for _ in range(frames):
            res = self.pipeline.detect(img)
            ring_state, _rc = self.ring_stable.feed(res.ring_state)
            a_present, _ac = self.a_stable.feed(res.a_present)
            before = self.sm.state
            state, event, message = self.sm.feed(ring_state, a_present)
            kept = (res, ring_state, a_present, before, state, event, message)
            if event != Event.NONE:
                break
        res, ring_state, a_present, before, state, event, message = kept
        self.last_result = res
        self._last_event = event
        self._last_message = message

        auto_verdict = self._auto_verdict(state, event)
        prev_state = self.records[-1].state_after if self.records else ""
        rec = FlowRecord(
            index=self.cursor + 1,
            filename=os.path.basename(path),
            source=path,
            ring_state=ring_state,
            ring_present_score=float(getattr(res, "ring_present_score", 0.0)),
            ring_absent_score=float(getattr(res, "ring_absent_score", 0.0)),
            a_present=bool(a_present),
            a_score=float(getattr(res, "a_score", 0.0)),
            state_before=before.value,
            state_after=state.value,
            event=event.value,
            node=self._business_node(
                state.value, event.value, ring_state, bool(a_present),
                message, prev_state, first=not self.records),
            auto_verdict=auto_verdict,
            manual_verdict=PENDING,
            message=message,
            notes=getattr(res, "notes", "") or "",
            timestamp=self._now(),
        )
        rec.final_verdict = self._final_verdict(rec)
        self.records.append(rec)

        # 模拟实际操作：判到漏装（报警）后自动复位，后续照片从头判定，
        # 等价于真实产线上"报警 → 操作工复位 → 继续生产"。
        if rec.is_alarm or state == CycleState.FAULT:
            self.sm.reset()
            self.ring_stable.reset()
            self.a_stable.reset()
            rec.message = f"{rec.message}；已自动复位（模拟实际操作），后续照片重新判定"
        return rec

    @staticmethod
    def _business_node(state_after: str, event: str, ring_state: str,
                       a_present: bool, message: str, prev_state: str,
                       first: bool = False) -> str:
        """把一次判定映射成业务流程节点（演示核对用）。

        节点口径与流程图一致：安装结束/待机 → 无轴承 → 放置轴承 →
        安装结束/待机（闭环）；异常照片 → 报警/异常节点。
        """
        if "读取失败" in message:
            return "异常：照片读取失败"
        if event in FAULT_EVENTS:
            return FAULT_EVENTS[event]
        if event == "CYCLE_OK":
            return "安装结束/待机（检测到工件A，本件完成）"
        if event == "CYCLE_START":
            return "放置轴承（检测到轴承）"
        if state_after == "FAULT":
            return "报警：异常锁存"
        if state_after == "IDLE":
            # 安装结束/待机（合并态）
            if a_present:
                return "安装结束/待机（工件A在位，重复检测）"
            if prev_state == "RING_READY" and ring_state == "ABSENT":
                return "轴承被取走（未报警）"
            if ring_state == "UNCERTAIN":
                return "安装结束/待机（画面不确定）"
            return "安装结束/待机（等待检测到轴）"
        if state_after == "RING_READY":
            if ring_state == "ABSENT":
                return "轴承被取走（未报警）"
            if ring_state == "UNCERTAIN":
                return "放置轴承（画面不确定，等待工件A）"
            return "放置轴承（等待工件A）"
        return "安装结束/待机"

    def _auto_verdict(self, state: CycleState, event: Event) -> str:
        if state == CycleState.FAULT:
            return FAIL
        if state.value in (self.cfg.auto_pass_states or []):
            return PASS
        if event == Event.CYCLE_OK:
            return PASS
        return PENDING

    def _final_verdict(self, rec: FlowRecord) -> str:
        rule = (self.cfg.judge_rule or "hybrid").lower()
        if rule == "manual":
            return rec.manual_verdict if rec.manual_verdict in (PASS, FAIL) else PENDING
        if rule == "auto":
            return rec.auto_verdict
        # hybrid：人工优先，未判时用自动建议
        if rec.manual_verdict in (PASS, FAIL):
            return rec.manual_verdict
        return rec.auto_verdict

    def judge(self, index: int, verdict: str) -> Optional[FlowRecord]:
        """对第 index 条记录（1 起）设置人工判定，重算最终判定。"""
        if not (1 <= index <= len(self.records)):
            return None
        rec = self.records[index - 1]
        rec.manual_verdict = verdict if verdict in (PASS, FAIL) else PENDING
        rec.final_verdict = self._final_verdict(rec)
        return rec

    def judge_current(self, verdict: str) -> Optional[FlowRecord]:
        if self.records:
            return self.judge(len(self.records), verdict)
        return None

    def last_record(self) -> Optional[FlowRecord]:
        return self.records[-1] if self.records else None

    def set_rule(self, rule: str) -> None:
        """改判断规则后重算所有最终判定。"""
        self.cfg.judge_rule = rule
        for rec in self.records:
            rec.final_verdict = self._final_verdict(rec)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------ 报告
    def report_lines(self) -> List[str]:
        done, total, pct = self.progress()
        lines = [
            "=== 照片流程演示报告 ===",
            f"来源目录：{self.source_dir()}",
            f"目标文件夹：{self.target_dir()}",
            f"判断规则：{self.cfg.judge_rule}  自动通过状态：{self.cfg.auto_pass_states}",
            f"进度：{done}/{total}（{pct:.0f}%）",
            "",
            f"{'序号':<4}{'文件名':<26}{'流程节点':<24}{'自动':<8}{'人工':<8}{'最终':<8}{'状态变化':<28}{'事件'}",
        ]
        for r in self.records:
            change = f"{r.state_before}→{r.state_after}"
            lines.append(
                f"{r.index:<4}{r.filename[:24]:<26}{r.node[:22]:<24}{r.auto_verdict:<8}"
                f"{r.manual_verdict:<8}{r.final_verdict:<8}{change:<28}{r.event}"
            )
        p = sum(1 for r in self.records if r.final_verdict == PASS)
        f = sum(1 for r in self.records if r.final_verdict == FAIL)
        lines += ["", f"汇总：通过 {p} / 不通过 {f} / 未判定 {len(self.records) - p - f}"]
        return lines

    def report_text(self) -> str:
        return "\n".join(self.report_lines())

    def export_report(self, path: str) -> bool:
        """导出 CSV（utf-8-sig，Excel 可直接打开）。"""
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["序号", "文件名", "流程节点", "自动判定", "人工判定", "最终判定",
                            "环状态", "有环NCC", "无环NCC", "A到位", "A得分",
                            "状态前", "状态后", "事件", "说明", "备注", "时间"])
                for r in self.records:
                    w.writerow([r.index, r.filename, r.node, r.auto_verdict, r.manual_verdict,
                                r.final_verdict, r.ring_state,
                                round(r.ring_present_score, 3), round(r.ring_absent_score, 3),
                                "是" if r.a_present else "否", round(r.a_score, 3),
                                r.state_before, r.state_after, r.event,
                                r.message, r.notes, r.timestamp])
            return True
        except Exception:
            return False
