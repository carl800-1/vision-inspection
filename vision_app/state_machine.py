# -*- coding: utf-8 -*-
"""事件驱动防错状态机（不使用时间窗口）。

三个正常状态对应三张照片：
    1 = 无环、无A        → IDLE（周期未开始）
    2 = 有环、无A        → RING_READY（周期开始并锁存）
    3 = A出现（环被遮挡） → CYCLE_DONE（本件合格，周期结束）

异常：
    A 在 IDLE 出现               → FAULT（漏放环，停机检查）
    RING_READY 时环在A前消失     → FAULT（环被取走，可选）
    FAULT 需人工复位
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class CycleState(Enum):
    IDLE = "IDLE"                     # 状态1：等待环
    RING_READY = "RING_READY"         # 状态2：已锁存有环
    CYCLE_DONE = "CYCLE_DONE"         # 状态3：A到位，本周期结束
    FAULT = "FAULT"                   # 异常锁存


class Event(Enum):
    NONE = "NONE"
    CYCLE_START = "CYCLE_START"
    CYCLE_OK = "CYCLE_OK"
    FAULT_A_FIRST = "FAULT_A_FIRST"
    FAULT_RING_REMOVED = "FAULT_RING_REMOVED"
    RESET = "RESET"


@dataclass
class StableSignal:
    """连续 N 帧一致才输出，避免单帧抖动误触发。"""
    confirm: int = 3
    value = None
    candidate = None
    count: int = 0

    def feed(self, value):
        if value == self.value:
            self.candidate, self.count = None, 0
            return self.value, False
        if value == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = value, 1
        if self.confirm <= 1 or self.count >= self.confirm:
            old = self.value
            self.value = value
            self.candidate, self.count = None, 0
            return self.value, old != value
        return self.value, False

    def reset(self):
        self.value = self.candidate = None
        self.count = 0


class PokerStateMachine:
    def __init__(self, ring_removal_is_fault: bool = True):
        self.state = CycleState.IDLE
        self.ring_removal_is_fault = ring_removal_is_fault
        self.ok_count = 0
        self.fault_count = 0
        self.last_event = Event.NONE
        self.last_message = "等待环放置（状态1）"

    def reset(self):
        self.state = CycleState.IDLE
        self.last_event = Event.RESET
        self.last_message = "已复位，等待环放置"

    def feed(self, ring_state: str, a_present: bool) -> Tuple[CycleState, Event, str]:
        """输入已去抖的环状态与A状态，返回 (状态, 事件, 提示)。"""
        self.last_event = Event.NONE
        self.last_message = ""

        if self.state == CycleState.FAULT:
            self.last_message = "异常已锁存，请人工复位后继续"
            return self.state, self.last_event, self.last_message

        if self.state == CycleState.IDLE:
            if a_present:
                self.state = CycleState.FAULT
                self.fault_count += 1
                self.last_event = Event.FAULT_A_FIRST
                self.last_message = "异常：工件A先于环出现（漏放环），请停机检查"
            elif ring_state == "PLACED":
                self.state = CycleState.RING_READY
                self.last_event = Event.CYCLE_START
                self.last_message = "周期开始：已检测到环（状态2），等待工件A"
            else:
                self.last_message = "等待环放置（状态1）"

        elif self.state == CycleState.RING_READY:
            if a_present:
                self.state = CycleState.CYCLE_DONE
                self.ok_count += 1
                self.last_event = Event.CYCLE_OK
                self.last_message = "OK：工件A到位，本周期结束（状态3）"
            elif ring_state == "ABSENT" and self.ring_removal_is_fault:
                self.state = CycleState.FAULT
                self.fault_count += 1
                self.last_event = Event.FAULT_RING_REMOVED
                self.last_message = "异常：环在工件A到位前被取走"
            elif ring_state == "ABSENT":
                self.state = CycleState.IDLE
                self.last_message = "环已取走，回到等待状态（未计OK）"
            else:
                self.last_message = "环已锁存，等待工件A（状态2）"

        elif self.state == CycleState.CYCLE_DONE:
            # A 撤走且环也不在 → 允许下一件；A 仍在则保持完成态
            if not a_present and ring_state == "ABSENT":
                self.state = CycleState.IDLE
                self.last_message = "工件已取下，等待下一件（状态1）"
            else:
                self.last_message = "本件已完成（状态3），等待取件"

        return self.state, self.last_event, self.last_message

    @property
    def state_index(self) -> int:
        return {
            CycleState.IDLE: 1,
            CycleState.RING_READY: 2,
            CycleState.CYCLE_DONE: 3,
            CycleState.FAULT: 0,
        }[self.state]
