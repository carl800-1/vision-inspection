# -*- coding: utf-8 -*-
"""事件驱动防错状态机（不使用时间窗口）。

检测事件与显示状态（安装结束与待机合并为一个状态）：
    检测到轴（无环、无A）   → 安装结束/待机（IDLE，自保持）
    检测到轴承（有环、无A） → RING_READY（显示「放置轴承」，锁存）
    检测到工件A（环被遮挡） → 本件完成，回到 安装结束/待机（IDLE）

异常（漏装，需人工复位）：
    无轴承态（检测到轴之后）出现工件A → FAULT（A先于轴承，漏放环）
    RING_READY 时环在A前消失          → FAULT（环被取走，可选）
    FAULT 需人工复位
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class CycleState(Enum):
    IDLE = "IDLE"                     # 安装结束/待机（合并态）：等待检测到轴
    RING_READY = "RING_READY"         # 放置轴承：已锁存有环，等待工件A
    FAULT = "FAULT"                   # 异常锁存（漏装）


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
        self.await_removal = False   # 刚完成的工件尚未取下（重复检测自保持）
        self.last_event = Event.NONE
        self.last_message = "安装结束/待机：等待检测到轴或轴承"

    def reset(self):
        self.state = CycleState.IDLE
        self.await_removal = False
        self.last_event = Event.RESET
        self.last_message = "已复位，安装结束/待机"

    def feed(self, ring_state: str, a_present: bool) -> Tuple[CycleState, Event, str]:
        """输入已去抖的环状态与A状态，返回 (状态, 事件, 提示)。"""
        self.last_event = Event.NONE
        self.last_message = ""

        if self.state == CycleState.FAULT:
            self.last_message = "异常已锁存，请人工复位后继续"
            return self.state, self.last_event, self.last_message

        if self.state == CycleState.IDLE:
            # 安装结束/待机（合并态）
            if a_present:
                if self.await_removal:
                    self.last_message = "安装结束/待机：工件A仍在位（重复检测），保持待机"
                else:
                    self.state = CycleState.FAULT
                    self.fault_count += 1
                    self.last_event = Event.FAULT_A_FIRST
                    self.last_message = "漏装：工件A先于轴承出现（顺序异常），请停机检查"
            else:
                self.await_removal = False   # 工件已取下，下一件从头计
                if ring_state == "PLACED":
                    self.state = CycleState.RING_READY
                    self.last_event = Event.CYCLE_START
                    self.last_message = "检测到轴承 → 放置轴承，等待工件A"
                elif ring_state == "ABSENT":
                    self.last_message = "安装结束/待机：检测到轴（无轴承），等待放置轴承"
                else:
                    self.last_message = "安装结束/待机：画面不确定，保持待机"

        elif self.state == CycleState.RING_READY:
            if a_present:
                self.ok_count += 1
                self.state = CycleState.IDLE
                self.await_removal = True
                self.last_event = Event.CYCLE_OK
                self.last_message = "OK：检测到工件A，安装结束 → 安装结束/待机"
            elif ring_state == "ABSENT" and self.ring_removal_is_fault:
                self.state = CycleState.FAULT
                self.fault_count += 1
                self.last_event = Event.FAULT_RING_REMOVED
                self.last_message = "漏装：轴承在工件A到位前被取走"
            elif ring_state == "ABSENT":
                self.state = CycleState.IDLE
                self.last_message = "轴承被取走，回到安装结束/待机（未计OK）"
            else:
                self.last_message = "放置轴承：环已锁存，等待工件A"

        return self.state, self.last_event, self.last_message

    @property
    def state_index(self) -> int:
        return {
            CycleState.IDLE: 1,
            CycleState.RING_READY: 2,
            CycleState.FAULT: 0,
        }[self.state]
