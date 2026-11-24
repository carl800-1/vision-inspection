# -*- coding: utf-8 -*-
"""相机连通性诊断工具（MV-CA050-10GC / 海康 MVS GigE）。

分步验证：SDK 环境 → 枚举 → 连接 → 参数 → 单帧 → 连续采集 → 判定特征值 → 断开。
本机 MVS 不完整时给出明确提示，不崩溃；目标机上运行可验证取流全链路。

用法：
    CameraProbe.exe                 自动枚举，取 5 帧
    CameraProbe.exe --ip 192.168.1.10 --frames 10
    CameraProbe.exe --no-detect     只测取流，不跑判定
"""
import os
import sys
import time
from datetime import datetime

OUT_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.join(OUT_DIR, "probe_captures")
LOG_PATH = os.path.join(OUT_DIR, "camera_probe.log")

_lines = []


def out(msg=""):
    print(msg)
    _lines.append(str(msg))
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
    except Exception:
        pass


def prepare_mvs_dll_path():
    """把 MVS 运行时 DLL 目录加进 PATH / add_dll_directory。

    MvCameraControl_class.py 用裸名 WinDLL("MvCameraControl.dll") 加载，
    依赖 Windows DLL 搜索路径。实测完整 MVS 客户端把 DLL 放在公共目录：
      C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64
    而不是 MVS 安装目录下的 Development\\Bin（那里只有示例 exe）。
    这里同时覆盖公共目录、旧版 SDK 目录和自定义 MVCAM_SDK_PATH。
    """
    common = os.environ.get("CommonProgramFiles(x86)",
                            r"C:\Program Files (x86)\Common Files")
    sdk = os.environ.get("MVCAM_SDK_PATH", r"C:\Program Files (x86)\MVS")
    candidates = [
        # 关键：完整 MVS 客户端的运行时 DLL 实际位置
        os.path.join(common, "MVS", "Runtime", "Win64_x64"),
        os.path.join(common, "MVS", "Runtime", "Win32_i86"),
        # 兼容旧版 SDK 布局与自定义路径
        os.path.join(sdk, "Runtime", "x64"),
        os.path.join(sdk, "Runtime", "Win64_x64"),
        os.path.join(sdk, "Development", "Bin", "Win64"),
        os.path.join(sdk, "Development", "Bin", "win64"),
        os.path.join(sdk, "Development", "Samples", "Python", "MvImport"),
    ]
    added = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        added.append(d)
    return added


def section(title):
    out("\n" + "=" * 62)
    out("  " + title)
    out("=" * 62)


def result(ok, label, detail=""):
    mark = "PASS" if ok else "FAIL"
    out(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def parse_args():
    ip, frames, no_detect = None, 5, False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--ip" and i + 1 < len(args):
            ip = args[i + 1]
            i += 2
        elif args[i] == "--frames" and i + 1 < len(args):
            try:
                frames = max(1, min(60, int(args[i + 1])))
            except ValueError:
                pass
            i += 2
        elif args[i] == "--no-detect":
            no_detect = True
            i += 1
        else:
            i += 1
    return ip, frames, no_detect


def enum_ips_via_child():
    """用独立子进程枚举，绕过 MVS SDK '枚举后无法直连' 的 bug。

    用环境变量 _PROBE_ENUM_CHILD 标记子进程模式（比命令行参数在
    PyInstaller onefile 下更可靠）。子进程只做枚举、输出 IP 后立刻退出，
    绝不跑完整诊断流程，避免递归。
    """
    import subprocess
    env = os.environ.copy()
    env["_PROBE_ENUM_CHILD"] = "1"
    cmd = ([sys.executable, os.path.abspath(__file__)]
           if not getattr(sys, "frozen", False) else [sys.executable])
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            # 只接受形似 IP 的 token，杜绝把诊断文本误当 IP
            import re
            return [t for t in r.stdout.split()
                    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", t)]
    except Exception:
        pass
    return []


def main():
    # 子进程模式：只做枚举，输出 IP 后退出（供 enum_ips_via_child 调用）。
    # 放在最开头，先于任何 section()/out() 输出，确保 onefile 下也不递归。
    if os.environ.get("_PROBE_ENUM_CHILD") == "1" or "--_enum_child" in sys.argv:
        try:
            prepare_mvs_dll_path()
            from hik_camera import HikCamera
            print(" ".join(HikCamera._get_dev_info().keys()))
            sys.exit(0)
        except Exception:
            sys.exit(1)

    ip_arg, frames, no_detect = parse_args()
    section("相机连通性诊断 · MV-CA050-10GC")
    out(f"  时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out(f"  输出目录: {OUT_DIR}")
    out(f"  取帧数: {frames}   判定: {'关闭' if no_detect else '开启'}")

    # ---- 1. SDK 环境 ----
    section("1 / 7  MVS SDK 环境检查")
    sdk = os.environ.get("MVCAM_SDK_PATH", r"C:\Program Files (x86)\MVS")
    common = os.environ.get("CommonProgramFiles(x86)",
                            r"C:\Program Files (x86)\Common Files")
    out(f"  MVS 安装目录 = {sdk}")
    out(f"  公共运行时目录 = {os.path.join(common, 'MVS', 'Runtime')}")
    dll_candidates = [
        os.path.join(common, "MVS", "Runtime", "Win64_x64", "MvCameraControl.dll"),
        os.path.join(common, "MVS", "Runtime", "Win32_i86", "MvCameraControl.dll"),
        os.path.join(sdk, "Runtime", "x64", "MvCameraControl.dll"),
        os.path.join(sdk, "Development", "Bin", "Win64", "MvCameraControl.dll"),
        os.path.join(sdk, "Development", "Bin", "win64", "MvCameraControl.dll"),
    ]
    found = [p for p in dll_candidates if os.path.exists(p)]
    if found:
        result(True, "MvCameraControl.dll 已找到", found[0])
    else:
        result(False, "MvCameraControl.dll 未找到",
               "请安装完整 MVS 客户端（含 Common Files\\MVS\\Runtime 运行时）")
        out("  若已安装仍报此错，检查环境变量 MVCAM_SDK_PATH")
        out("  指向 MVS 安装根目录，例如 C:\\Program Files (x86)\\MVS")

    # ---- 2. 导入 SDK ----
    section("2 / 7  加载相机驱动")
    paths = prepare_mvs_dll_path()
    if paths:
        out(f"  已加入 DLL 搜索路径: {len(paths)} 处")
        for d in paths:
            out(f"    {d}")
    try:
        from hik_camera import HikCamera
        result(True, "hik_camera 导入成功")
    except Exception as e:
        result(False, "hik_camera 导入失败", f"{type(e).__name__}: {e}")
        out("  常见原因：DLL 缺失 / MVS 版本不匹配 / Python 位数与 SDK 不一致")
        section("诊断中止")
        out("  SDK 未就绪，后续步骤无法执行。请按上面提示处理后重试。")
        return 1

    # ---- 3. 枚举 ----
    # 指定 --ip 时跳过枚举：既绕开 MVS SDK“枚举后无法直连”的 bug，
    # 也避免打包后子进程枚举的开销；未指定才枚举。
    section("3 / 7  枚举相机")
    ips = []
    if ip_arg:
        result(True, "跳过枚举（已用 --ip 指定目标）", ip_arg)
        ips = [ip_arg]
    else:
        ips = enum_ips_via_child()
        if ips:
            result(True, f"发现 {len(ips)} 台相机", ", ".join(ips))
        else:
            # 回退：同进程直接调 SDK 枚举（用 _get_dev_info，不用 get_all_ips——
            # 后者打包后会用 sys.executable 重新拉起 exe 导致无限递归）
            try:
                ips = list(HikCamera._get_dev_info().keys())
                if ips:
                    result(True, f"发现 {len(ips)} 台相机（直连枚举）", ", ".join(ips))
                else:
                    result(False, "未枚举到相机",
                           "检查网线/PoE供电、相机与主机同网段、防火墙是否拦 GigE")
            except Exception as e:
                result(False, "枚举异常", f"{type(e).__name__}: {e}")
                out("  枚举失败时可用 --ip 参数手动指定相机 IP 继续")

    target = ip_arg or (ips[0] if ips else None)
    if not target:
        out("\n  没有可用 IP。用法示例：CameraProbe.exe --ip 192.168.1.10")
        section("诊断中止")
        return 1
    out(f"  目标相机: {target}")

    # ---- 4. 连接 ----
    section("4 / 7  连接相机")
    cam = None
    try:
        cam = HikCamera(target)
        cam.__enter__()
        result(True, "相机已打开", f"IP={target}")
    except Exception as e:
        result(False, "打开相机失败", f"{type(e).__name__}: {e}")
        out("  排查：IP 是否被占用（其它程序独占）、相机是否被 reset 中、"
            " host_ip 是否可达")
        section("诊断中止")
        return 1

    # ---- 5. 参数 ----
    section("5 / 7  关键参数读取")
    keys = ["ExposureAuto", "ExposureTime", "Gain", "PixelFormat",
            "TriggerMode", "TriggerSource", "Width", "Height",
            "AcquisitionFrameRateEnable"]
    for k in keys:
        try:
            out(f"  {k:<28} = {cam[k]}")
        except Exception as e:
            out(f"  {k:<28} = <读取失败: {e}>")
    out("  提示：hik_camera 打开时默认 软触发 + 自动曝光；")
    out("        产线判定建议固定 ExposureAuto=Off 并标定曝光值。")

    # ---- 6. 取帧 ----
    section("6 / 7  单帧 / 连续采集")
    os.makedirs(CAP_DIR, exist_ok=True)
    saved = []
    for n in range(frames):
        t0 = time.time()
        try:
            img = cam.robust_get_frame()
        except Exception as e:
            result(False, f"第 {n+1} 帧取流失败", f"{type(e).__name__}: {e}")
            break
        dt = (time.time() - t0) * 1000
        if img is None:
            result(False, f"第 {n+1} 帧返回空", "检查触发模式与 SDK 取流超时")
            break
        path = os.path.join(CAP_DIR, f"frame_{n+1:02d}_{datetime.now():%H%M%S}.png")
        try:
            import cv2
            cv2.imwrite(path, img)
            saved.append(path)
        except Exception:
            pass
        result(True, f"第 {n+1}/{frames} 帧",
               f"{img.shape[1]}x{img.shape[0]}  {dt:.0f} ms  -> {os.path.basename(path)}")
    if saved:
        result(True, f"共保存 {len(saved)} 帧", CAP_DIR)

    # ---- 7. 判定特征值 ----
    if not no_detect and saved:
        section("7 / 7  判定特征值（用于现场标定）")
        try:
            sys.path.insert(0, OUT_DIR)
            from ring_detect_core import detect_ring
            import cv2
            for p in saved:
                img = cv2.imread(p)
                if img is None:
                    continue
                r = detect_ring(img)
                out(f"  {os.path.basename(p)}: 判定={r['verdict']}  "
                    f"暗谷={r['valley']:.0f}@r={r['valley_r']}px  "
                    f"亮峰={r['peak']:.0f}  落差={r['contrast']:.0f}  "
                    f"裕度={r['margin']:.2f}")
            out("  注意：阈值 55 是在变焦特写上标定的。若现场取景比例/打光不同，")
            out("        请以实测暗谷值为依据重新标定，不要直接套用 55。")
        except Exception as e:
            result(False, "判定模块不可用", f"{type(e).__name__}: {e}")
            out("  需把 ring_detect_core.py 放在本工具同目录。")
    elif no_detect:
        section("7 / 7  判定已跳过（--no-detect）")

    # ---- 断开 ----
    section("完成情况")
    try:
        cam.__exit__(None, None, None)
        result(True, "相机已正常关闭")
    except Exception as e:
        result(False, "关闭相机异常", str(e))
    out(f"  日志: {LOG_PATH}")
    return 0 if saved else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        out(f"\n[未捕获异常] {type(e).__name__}: {e}")
        import traceback
        out(traceback.format_exc())
        code = 2
    out("\n按回车键退出...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)
