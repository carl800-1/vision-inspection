# -*- coding: utf-8 -*-
"""自学习训练命令行入口。

用法示例：
  # 1) 从标准目录训练（推荐长期收集照片）
  python train_model.py --dataset dataset --config config.json

  # 2) 直接用三张状态图训练（1=无环 2=有环 3=放A）
  python train_model.py --state1 state_1_core.jpg --state2 state_2_core.jpg \
      --state3 state_3_core.jpg --config config.json

  # 3) 只生成默认配置文件
  python train_model.py --init-config config.json

训练结果：模型 JSON + model/templates/*.png，并会回写 config.json 中
自动推导出的 valley_th 等参数。
"""
from __future__ import annotations

import argparse
import os
import sys

from vision_app.config import AppConfig, load_config, save_config
from vision_app.dataset import DatasetManager
from vision_app.trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description="环形轴承防错检测 — 照片自学习训练")
    p.add_argument("--config", default=os.path.join(HERE, "config.json"),
                   help="配置文件路径，默认 ./config.json")
    p.add_argument("--dataset", default="", help="训练照片根目录")
    p.add_argument("--state1", nargs="*", default=[], help="无环照片（可多张）")
    p.add_argument("--state2", nargs="*", default=[], help="有环照片（可多张）")
    p.add_argument("--state3", nargs="*", default=[], help="放工件A照片（可多张）")
    p.add_argument("--model", default="", help="模型输出路径，默认取配置中的 paths.model")
    p.add_argument("--init-config", default="", help="只生成默认配置文件并退出")
    p.add_argument("--ring-roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="环模板 ROI，归一化四数")
    p.add_argument("--a-roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="工件A模板 ROI，归一化四数")
    p.add_argument("--a-hole-roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="A偏心孔搜索区，归一化四数")
    p.add_argument("--a-edge-roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="A横边搜索区，归一化四数")
    p.add_argument("--center-roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="中心轴/环遮挡检测区，归一化四数")
    p.add_argument("--ring-center", nargs=2, type=float, metavar=("X", "Y"),
                   help="环心种子（归一化0~1），不填则训练时自动定位")
    p.add_argument("--crop-half", type=int, default=0,
                   help="以环心为中心的裁剪半径（像素），0 表示用配置值")
    args = p.parse_args(argv)

    if args.init_config:
        save_config(AppConfig(), args.init_config)
        print(f"默认配置已写入：{args.init_config}")
        return 0

    cfg = load_config(args.config)
    for arg_name, field_name in (
        ("ring_roi", "ring_roi"), ("a_roi", "a_roi"),
        ("a_hole_roi", "a_hole_roi"), ("a_edge_roi", "a_edge_roi"),
        ("center_roi", "center_roi"),
    ):
        value = getattr(args, arg_name)
        if value:
            setattr(cfg.detect, field_name, list(value))
    if args.ring_center:
        cfg.detect.ring_center = list(args.ring_center)
    if args.crop_half:
        cfg.detect.crop_half = args.crop_half
    model_path = args.model or os.path.join(HERE, cfg.paths.model)
    if not os.path.isabs(model_path):
        model_path = os.path.join(HERE, model_path)

    trainer = Trainer(cfg.detect)
    n_dataset = 0
    dataset_root = args.dataset or os.path.join(HERE, cfg.paths.dataset)
    if os.path.isdir(dataset_root):
        found = DatasetManager(dataset_root).scan()
        for label, paths in found.items():
            for path in paths:
                n_dataset += int(trainer.add_image(path, label))

    explicit = sum(trainer.add_image(p, "state1") for p in args.state1)
    explicit += sum(trainer.add_image(p, "state2") for p in args.state2)
    explicit += sum(trainer.add_image(p, "state3") for p in args.state3)

    counts = trainer.counts()
    print(f"导入照片：目录 {n_dataset} 张 + 命令行 {explicit} 张")
    print(f"分类统计：{counts}")
    if counts.get("absent", 0) == 0 or counts.get("present", 0) == 0:
        print("错误：至少需要 state1(无环) 和 state2(有环) 各一张照片。", file=sys.stderr)
        return 2

    model = trainer.train(model_path, cfg.detect)
    save_config(cfg, args.config)
    print("\n".join(trainer.report))
    print(f"\n模板：present={len(model.templates.get('present', []))} "
          f"absent={len(model.templates.get('absent', []))} "
          f"a={len(model.templates.get('a', []))}")
    print(f"配置已更新：{args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
