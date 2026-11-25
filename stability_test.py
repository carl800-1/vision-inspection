# -*- coding: utf-8 -*-
"""自动定位环心稳定性测试：固定环心 vs 自动定位，含亮度/平移扰动。"""
import os
import sys
import cv2
import numpy as np
from vision_app.config import load_config
from vision_app.model import Model
from vision_app.detectors import VisionPipeline
from vision_app.imageio import imread_unicode

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "new_imgs")
cfg = load_config(os.path.join(HERE, "config.json"))
model = Model.load(os.path.join(HERE, "model", "ring_model.json"))
d = cfg.detect


def run(tag, imgs, use_locator):
    d.use_locator = use_locator
    p = VisionPipeline(d, model)
    print(f"\n=== {tag} (use_locator={use_locator}) ===")
    for i, img in enumerate(imgs, 1):
        r = p.detect(img)
        g = r.geometry
        print(f"state{i}: ring={r.ring_state:9s} A={str(r.a_present):5s} "
              f"环心=({g.get('cx')},{g.get('cy')}) ok={g.get('locator_ok')} "
              f"score={g.get('locator_score', 0):.2f}")


base = [imread_unicode(os.path.join(IMG_DIR, f"{i}.jpg"))
        for i in (1, 2, 3)]

run("原始三态-固定环心(基线)", base, False)
run("原始三态-自动定位", base, True)

for k in (0.75, 1.25):
    pert = [np.clip(im.astype(np.float32) * k, 0, 255).astype(np.uint8)
            for im in base]
    run(f"亮度x{k}-自动定位", pert, True)

for dx, dy in ((25, 0), (0, 25), (-25, -25)):
    pert = []
    for im in base:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        pert.append(cv2.warpAffine(im, M, (im.shape[1], im.shape[0])))
    run(f"平移({dx},{dy})-自动定位", pert, True)
