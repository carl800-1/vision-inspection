# vision-inspection

环形轴承防错视觉检测系统。基于 OpenCV + PySide6，支持海康工业相机（MV-CA050-10GC）接入、实时判定、照片自学习。当前版本 **1.0.1**。

## 判定逻辑（事件驱动，不用时间窗口）

三态状态机，对应三个工位状态：

| 状态 | 画面 | 判定 | 状态机 |
|---|---|---|---|
| 状态1 | 无环、无工件A | 环 ABSENT | `IDLE` 等待环 |
| 状态2 | 有环、无工件A | 环 PLACED，锁存 | `RING_READY` 周期开始 |
| 状态3 | 工件A到位遮挡环 | A 到位、中心清空 | `CYCLE_DONE` 本件 OK |

异常（进 FAULT，需人工复位）：
- A 先于环出现（漏放环）
- 环在 A 到位前被取走

每帧处理链：原图等比缩放到 1200px（**不裁剪**）→ 灰度 + 中值滤波 → 归一化 ROI → 环用径向暗谷亮度法、A 用偏心孔+横边+模板三特征投票 → 连续帧去抖（环 3 帧 / A 2 帧）→ 状态机。

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

相机尚未搭建，逻辑与界面已定稿，参数用三张现场图做过模拟验证（OK=1、异常=0）。**阈值与 ROI 为占位值，需相机装好后用同视野实拍照片重新标定**：

1. 设置页把图源改为 `hik`，填相机 IP，固定曝光/增益（关自动曝光）。
2. 同视野拍状态1/2/3 三张，导入"照片自学习"页训练更新基准。
3. 先用"打开图片判定"确认暗谷、A 得分、中心清空三项分离干净，再开连续采集。

关键参数在 `config.json`：`detect.valley_th`（暗谷阈值）、`a_feature_mode` 与 `a_min_votes`（工件A投票）、`center_edge_th` + `require_center_clear`（中心遮挡）、各 ROI 归一化坐标、`ring_confirm_frames` / `a_confirm_frames`（去抖帧数）。

## 标注图说明

模拟标注图上的圆圈是调试可视化，不参与判定：红色=霍夫圆候选（含误检）、蓝色=暗谷半径、黄色=亮峰半径、绿色点=选定圆心、粉色=工件A偏心孔。判定只依赖暗谷亮度与 A 特征投票。
