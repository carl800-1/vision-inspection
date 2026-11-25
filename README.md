# vision-inspection

环形轴承防错视觉检测系统。基于 OpenCV + PySide6，支持海康工业相机（MV-CA050-10GC）接入、实时判定、照片自学习。当前版本 **1.0.2**。

## 判定逻辑（事件驱动，不用时间窗口）

三态状态机，对应三个工位状态：

| 状态 | 画面 | 判定 | 状态机 |
|---|---|---|---|
| 状态1 | 无环、无工件A | 环 ABSENT | `IDLE` 等待环 |
| 状态2 | 有环、无工件A | 环 PLACED，锁存 | `RING_READY` 周期开始 |
| 状态3 | 工件A到位遮挡环 | A 到位（环被挡，判 UNCERTAIN 属正常） | `CYCLE_DONE` 本件 OK |

异常（进 FAULT，需人工复位）：
- A 先于环出现（漏放环）
- 环在 A 到位前被取走

**判定方法：裁剪模板法（`ring_mode=crop_template`）**。旧的"径向暗谷 + 霍夫找圆心"方案已弃用——霍夫在整图上会找到几百个候选圆，取"离中心最近"常落在螺栓孔上，导致暗谷量到外圈阴影、判错状态。新方案：

每帧处理链：原图等比缩放到 1200px → 以训练标定的环心为中心裁一块固定小图（默认半径 130px）→ 该小图分别与"有环 / 无环 / 工件A"三类示教模板做 NCC 匹配 → 取最高分类别 → 连续帧去抖（环 3 帧 / A 2 帧）→ 状态机。相机固定后环心不变，默认直接用固定环心（`use_locator=false`，最稳）；如相机可能轻微偏移，可开 `use_locator=true` 在环心附近做局部微调搜索。

新三张现场图（1920×1440 竖版）实测 NCC 分离度：

| 输入 | 有环模板 | 无环模板 | A模板 | 判定 |
|---|---|---|---|---|
| 状态1 无环 | 0.46 | **1.00** | 0.00 | ABSENT |
| 状态2 有环 | **1.00** | 0.42 | 0.00 | PLACED |
| 状态3 放A | 0.00 | 0.00 | **1.00** | A 到位 |

## 目录结构

```
ring_inspector_v2.py   主程序 GUI（检测 / 照片自学习 / 参数设置 三个页签）
train_model.py         命令行训练脚本
camera_probe.py        相机诊断工具（海康 MVS）
ring_detect_core.py    早期几何法核心（保留参考）
vision_app/            核心模块
  config.py            配置数据类与读写
  imageio.py           中文路径图像读写、ROI 工具
  model.py             示教模型存取
  detectors.py         环 / 工件A 检测器
  trainer.py           照片训练
  state_machine.py     事件驱动状态机
  camera.py            海康 / 演示 / 文件图源
  dataset.py           样本目录管理
simulate_1_2_3.py      三态模拟脚本
stability_test.py      定位稳定性测试（光照/平移扰动）
config.json            全部参数（阈值、ROI、相机）
model/ring_model.json  示教模型（模板图不入库，实拍后重新训练生成）
demo_images/           内置演示图
app_icon.ico           程序图标
RingInspector.spec     PyInstaller 打包配置
```

## 运行

```bash
pip install opencv-python numpy PySide6
python ring_inspector_v2.py              # 正常启动
python ring_inspector_v2.py --selftest   # 无相机自检（5 秒后写 selftest_v2.png/txt）
python train_model.py --init-config config.json
python simulate_1_2_3.py                 # 用三张状态图跑端到端模拟
```

## 当前阶段

相机尚未搭建，逻辑与界面已定稿。参数已用三张现场图（不入库）训练并端到端验证：`IDLE → RING_READY(CYCLE_START) → CYCLE_DONE(CYCLE_OK)`，OK=1、异常=0。**模板基于这三张图，相机装好后需用同视野实拍重新训练**：

1. 设置页把图源改为 `hik`，填相机 IP，固定曝光/增益（关自动曝光）。
2. 同视野拍状态1/2/3 三张，导入"照片自学习"页训练更新基准（自动定位环心，也可用 `--ring-center X Y` 手工指定）。
3. 先用"打开图片判定"确认三类 NCC 分离干净（各自模板≈1.0、其他<0.5），再开连续采集。

关键参数在 `config.json`：`detect.ring_mode=crop_template`、`ring_ncc_th` / `a_ncc_th`（NCC 阈值）、`crop_half`（裁剪半径）、`use_locator` / `locator_th`（是否微调环心）、`ring_center`（环心种子）、`ring_confirm_frames` / `a_confirm_frames`（去抖帧数）。旧的几何法参数（`valley_th` 等）仅作兼容保留，不再参与判定。

## 标注图说明

裁剪模板法的标注很干净：黄色方框=裁剪区（固定套住环）、绿色点=环心。不再有旧方案的满屏红圈（霍夫候选）、蓝/黄暗谷亮峰圈。状态3环被工件A遮挡时环判为 UNCERTAIN 属正常，此时由 A 模板判定周期结束。

## 重新训练

把同视野拍的三张状态图放到任意目录（如 `captures/`），按顺序执行：

```bash
# 自动定位环心
python train_model.py --state1 captures/1.jpg --state2 captures/2.jpg --state3 captures/3.jpg --config config.json
# 手工指定环心（归一化 x,y，相机固定后推荐，最稳）
python train_model.py --state1 captures/1.jpg --state2 captures/2.jpg --state3 captures/3.jpg --ring-center 0.5069 0.4896 --crop-half 130 --config config.json
python stability_test.py captures   # 可选：光照/平移扰动下的稳定性自测
```
