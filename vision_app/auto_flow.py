# -*- coding: utf-8 -*-
"""连接相机后的「自动拍照 → 判定 → 流程切换」控制器。

与 GUI 完全解耦：本模块只接收「有效照片的检测结果」「拍照/检测失败」「超时检查」
三类输入，输出状态、事件、显示文案和报警标志。这样既能被主界面 QTimer 驱动，
也能用 selftest_auto_flow.py 无相机、无界面地跑完所有分支。

业务规则（顺序处理，不用时间窗口，安装结束与待机合并为一个状态）：
    安装结束/待机  检测到轴（有主轴、无轴承）→ 「无轴承」；
                   未检测到轴 / 重复检测到工件A（上一件未取下）→ 自保持，不报警、无超时；
                   未拍到裸轴直接检出轴承 → 顺序提示，按「放置轴承」继续，不误报
    无轴承        下一张有效照片「有轴承」→ 「放置轴承」；
                   A 先出现 / 仍无轴承 → 漏装
    放置轴承      检测到工件A → 本件完成，回到「安装结束/待机」；
                   环在A到位前消失 → 漏装
    漏装          报警锁存，需人工复位

注：照片按时间逐张推进；「下一张」指下一张有效照片（取帧成功且检测成功）。
拍照失败、检测失败以及环状态 UNCERTAIN（且未检出A）都不计有效照片，重试直到超时。
检测到轴 = 有主轴 + 无轴承（上一件取走后主轴露出）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from .config import AutoFlowConfig


class AutoState(Enum):
    IDLE = "IDLE"                   # 安装结束/待机：等待检测到轴（合并态）
    WAIT_BEARING = "WAIT_BEARING"   # 无轴承：已检到轴，等待放置轴承
    BEARING_PLACED = "BEARING_PLACED"  # 放置轴承：已检到轴承，等待工件A
    MISSING = "MISSING"             # 漏装（报警锁存）


class AutoEvent(Enum):
    NONE = "NONE"
    FLOW_START = "FLOW_START"
    BEARING_OK = "BEARING_OK"
    CYCLE_OK = "CYCLE_OK"
    MISSING_BEARING = "MISSING_BEARING"
    PHOTO_TIMEOUT = "PHOTO_TIMEOUT"
    CAPTURE_FAIL = "CAPTURE_FAIL"
    DETECT_FAIL = "DETECT_FAIL"
    RESET = "RESET"


DISPLAY_TEXT = {
    AutoState.IDLE: "安装结束/待机",
    AutoState.WAIT_BEARING: "无轴承",
    AutoState.BEARING_PLACED: "放置轴承",
    AutoState.MISSING: "漏装",
}


@dataclass
class AutoRecord:
    """一条自动流程事件记录（可导出/显示在日志表格中）。"""
    index: int = 0
    photo_no: int = 0
    state: str = ""
    event: str = ""
    display: str = ""
    message: str = ""
    alarm: bool = False
    ring_state: str = ""
    shaft_present: bool = False
    a_present: bool = False
    timestamp: str = ""

    @staticmethod
    def now() -> str:
        return datetime.now().strftime("%H:%M:%S")


class AutoFlowController:
    """自动拍照检测流程状态机。

    闭环：安装结束/待机 →（检测到轴）→ 无轴承 →（检测到轴承）→ 放置轴承
          →（检测到工件A）→ 安装结束/待机。异常进入漏装报警锁存。
    """

    MAX_RECORDS = 300

    def __init__(self, cfg: AutoFlowConfig):
        self.cfg = cfg
        self.reset()

    # ------------------------------------------------------------ 基本控制
    def reset(self) -> AutoRecord:
        self.running = False
        self.state = AutoState.IDLE
        self.photo_no = 0
        self.bearing_streak = 0
        self.valid_count = 0
        self.invalid_streak = 0
        self.last_invalid_reason = ""
        self.deadline: Optional[float] = None
        self.records: List[AutoRecord] = []
        self.last_event = AutoEvent.RESET
        self.last_alarm = False
        return self._emit(AutoEvent.RESET, "已复位，等待启动", alarm=False)

    def start(self) -> AutoRecord:
        """启动自动流程（仍待「检测到轴」照片触发）。"""
        self.reset()
        self.running = True
        self.last_event = AutoEvent.NONE
        self.records = []  # reset 的 RESET 记录不进入本次日志
        return self._emit(AutoEvent.NONE, "自动流程已启动：等待「检测到轴」照片",
                          alarm=False)

    def stop(self) -> AutoRecord:
        self.running = False
        self.deadline = None
        return self._emit(AutoEvent.NONE, "自动流程已停止", alarm=False)

    @property
    def active(self) -> bool:
        return self.running and self.state in (
            AutoState.WAIT_BEARING, AutoState.BEARING_PLACED)

    @property
    def display(self) -> str:
        return DISPLAY_TEXT[self.state]

    def _emit(self, event: AutoEvent, message: str, *, alarm: bool = False,
              ring_state: str = "", shaft_present: bool = False,
              a_present: bool = False) -> AutoRecord:
        rec = AutoRecord(
            index=len(self.records) + 1,
            photo_no=self.photo_no,
            state=self.state.value,
            event=event.value,
            display=self.display,
            message=message,
            alarm=alarm,
            ring_state=ring_state,
            shaft_present=shaft_present,
            a_present=a_present,
            timestamp=AutoRecord.now(),
        )
        self.last_event = event
        self.last_alarm = alarm
        self.records.append(rec)
        if len(self.records) > self.MAX_RECORDS:
            self.records.pop(0)
            for i, r in enumerate(self.records, 1):
                r.index = i
        return rec

    def _fault(self, event: AutoEvent, message: str, **kwargs) -> AutoRecord:
        self.state = AutoState.MISSING
        self.deadline = None
        return self._emit(event, message, alarm=bool(self.cfg.alarm_enabled), **kwargs)

    # ------------------------------------------------------------ 有效照片
    def on_valid_photo(self, ring_state: str, shaft_present: bool = False,
                       a_present: bool = False, now: Optional[float] = None) -> AutoRecord:
        """输入一张有效照片的检测结果，返回本张产生的流程记录。"""
        now = time.monotonic() if now is None else now
        if not self.running:
            return self._emit(AutoEvent.NONE, "流程未启动，照片未参与判定",
                              ring_state=ring_state, shaft_present=shaft_present,
                              a_present=a_present)

        self.photo_no += 1
        self.valid_count += 1
        self.invalid_streak = 0
        self.last_invalid_reason = ""
        ring = str(ring_state or "UNCERTAIN").upper()
        shaft_ok = (not self.cfg.require_shaft) or bool(shaft_present)

        if self.state == AutoState.IDLE:
            # 安装结束/待机（合并态）：工件A在位一律自保持（重复检测），
            # 避免安装完成后的残留 A 被误判为新周期异常。
            if a_present:
                return self._emit(
                    AutoEvent.NONE, "安装结束/待机：工件A在位（重复检测），保持待机",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            if shaft_ok and ring == "ABSENT":
                # 检测到轴：上一件已取走，主轴露出 → 无轴承
                self.state = AutoState.WAIT_BEARING
                self.bearing_streak = 0
                self.deadline = now + self.cfg.photo_timeout_ms / 1000.0
                return self._emit(
                    AutoEvent.FLOW_START,
                    "检测到轴：有主轴、无轴承 → 无轴承，等待放置轴承",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            if ring == "PLACED":
                # 顺序提示：未拍到裸轴直接检出轴承（可能跳帧），按放置轴承继续，不误报
                self.state = AutoState.BEARING_PLACED
                self.bearing_streak = 1
                self.deadline = now + self.cfg.photo_timeout_ms / 1000.0
                return self._emit(
                    AutoEvent.BEARING_OK,
                    "顺序提示：未拍到裸轴直接检出轴承，按「放置轴承」继续，等待工件A",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            reason = ("未检出主轴，保持安装结束/待机" if not shaft_ok
                      else f"环状态={ring}，保持安装结束/待机")
            return self._emit(AutoEvent.NONE, reason, ring_state=ring,
                              shaft_present=shaft_present, a_present=a_present)

        if self.state == AutoState.WAIT_BEARING:
            self.deadline = now + self.cfg.photo_timeout_ms / 1000.0
            if a_present:
                return self._fault(
                    AutoEvent.MISSING_BEARING,
                    "漏装：工件A先于轴承出现（无轴承态检出A），请停机检查",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            if ring == "PLACED":
                self.state = AutoState.BEARING_PLACED
                self.bearing_streak = 1
                return self._emit(AutoEvent.BEARING_OK,
                                  "检测到轴承 → 放置轴承，等待工件A",
                                  ring_state=ring, shaft_present=shaft_present,
                                  a_present=a_present)
            return self._fault(
                AutoEvent.MISSING_BEARING,
                f"漏装：下一张有效照片未检出轴承（环状态={ring}）",
                ring_state=ring, shaft_present=shaft_present, a_present=a_present)

        if self.state == AutoState.BEARING_PLACED:
            self.deadline = now + self.cfg.photo_timeout_ms / 1000.0
            if a_present:
                self.state = AutoState.IDLE
                self.deadline = None
                return self._emit(
                    AutoEvent.CYCLE_OK,
                    "安装结束：检测到工件A，本件完成 → 安装结束/待机",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            if ring == "ABSENT":
                return self._fault(
                    AutoEvent.MISSING_BEARING,
                    "漏装：轴承在工件A到位前消失（环状态=ABSENT）",
                    ring_state=ring, shaft_present=shaft_present, a_present=a_present)
            if ring == "PLACED":
                self.bearing_streak += 1
            return self._emit(AutoEvent.NONE,
                              "放置轴承：等待工件A（画面持续观察中）",
                              ring_state=ring, shaft_present=shaft_present,
                              a_present=a_present)

        # MISSING：报警锁存，必须人工复位
        return self._emit(AutoEvent.NONE,
                          "漏装报警未复位：请检查后点「复位」重新开始",
                          ring_state=ring, shaft_present=shaft_present,
                          a_present=a_present)

    # ------------------------------------------------------------ 异常与超时
    def on_invalid(self, reason: str, *, event: AutoEvent = AutoEvent.CAPTURE_FAIL
                   ) -> Optional[AutoRecord]:
        """拍照失败 / 检测失败：不计有效照片，累计次数，超限判漏装。"""
        if not self.running or self.state == AutoState.MISSING:
            return None
        self.invalid_streak += 1
        self.last_invalid_reason = reason
        rec = self._emit(event, f"{reason}（第 {self.invalid_streak} 次，不计有效照片）")
        if (self.active and
                self.invalid_streak >= max(1, self.cfg.max_invalid_retries)):
            return self._fault(AutoEvent.PHOTO_TIMEOUT,
                               f"漏装：连续 {self.invalid_streak} 次无法取得有效照片（{reason}）")
        return rec

    def check_timeout(self, now: Optional[float] = None) -> Optional[AutoRecord]:
        """QTimer 每个拍照周期调用一次；等待下一张有效照片超过时限则报警。"""
        if not self.active or self.deadline is None:
            return None
        now = time.monotonic() if now is None else now
        if now < self.deadline:
            return None
        return self._fault(
            AutoEvent.PHOTO_TIMEOUT,
            f"漏装：启动后 {self.cfg.photo_timeout_ms} ms 内未取得下一张有效照片"
            + (f"（{self.last_invalid_reason}）" if self.last_invalid_reason else ""))

    # ------------------------------------------------------------ 报告
    def report_lines(self) -> List[str]:
        lines = [
            "=== 相机自动拍照检测流程 ===",
            f"状态：{self.display}  运行：{'是' if self.running else '否'}  "
            f"有效照片：{self.valid_count}  总拍照序号：{self.photo_no}",
            f"间隔：{self.cfg.interval_ms}ms  超时：{self.cfg.photo_timeout_ms}ms  "
            f"完成依据：检测到工件A",
            "",
        ]
        for r in self.records:
            prefix = "[报警]" if r.alarm else "     "
            lines.append(f"{prefix} [{r.timestamp}] #{r.photo_no} {r.state}/{r.event} "
                         f"→ {r.display}：{r.message}")
        return lines
