#!/usr/bin/env python3
# YOLO26 Inference Script (AXERARuntime)
# 2026-01-14 Version (modified)
# Copyright (c) 2026 D-Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) 2026 M5Stack Technology CO LTD
# Author: LittleMouse

import os
import cv2
import numpy as np
from time import time
import argparse
import logging
import axengine as ort

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YOLO26")


def infer_hw_layout(shape):
    shape = list(shape)
    if len(shape) == 4 and shape[-1] == 3:
        h = int(shape[1] or 640)
        w = int(shape[2] or 640)
        return h, w, "NHWC"
    if len(shape) == 4 and shape[1] == 3:
        h = int(shape[2] or 640)
        w = int(shape[3] or 640)
        return h, w, "NCHW"
    return 640, 640, "NCHW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', type=str, default='yolo26.onnx')
    ap.add_argument('--test-img', type=str, default='bus.jpg')
    ap.add_argument('--img-save-path', type=str, default='result_yolo26.jpg')
    ap.add_argument('--score-thres', type=float, default=0.25)
    ap.add_argument('--nms-thres', type=float, default=0.7)
    ap.add_argument('--providers', type=str, default='')
    opt = ap.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    t0 = time()
    providers = [p.strip() for p in opt.providers.split(",") if p.strip()] or None
    sess = ort.InferenceSession(opt.model_path, providers=providers)
    logger.debug(f"\033[1;31mLoad model time = {(time() - t0) * 1000:.2f} ms\033[0m")

    inp = sess.get_inputs()[0]
    input_name = inp.name
    m_h, m_w, layout = infer_hw_layout(inp.shape)

    img = cv2.imread(opt.test_img)
    if img is None:
        logger.error(f"Image not found or unreadable: {opt.test_img}")
        return

    t0 = time()
    orig_h, orig_w = img.shape[:2]
    scale = min(m_h / orig_h, m_w / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, 0, m_h - new_h, 0, m_w - new_w,
        cv2.BORDER_CONSTANT, value=(127, 127, 127)
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_tensor = rgb[None, ...].astype(np.uint8) if layout == "NHWC" else np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.uint8)
    logger.debug(f"\033[1;31mPre-process time = {(time() - t0) * 1000:.2f} ms\033[0m")

    t0 = time()
    ort_outputs = sess.run(None, {input_name: input_tensor})
    out_metas = sess.get_outputs()
    logger.debug(f"\033[1;31mForward time = {(time() - t0) * 1000:.2f} ms\033[0m")

    t0 = time()
    strides = (8, 16, 32)
    conf_raw = -np.log(1 / opt.score_thres - 1)
    detections = []

    output_items = []
    for meta, data in zip(out_metas, ort_outputs):
        shape = list(meta.shape)
        if any(s is None or isinstance(s, str) for s in shape):
            shape = list(data.shape)
        output_items.append((data, shape))

    for stride in strides:
        grid_size = m_h // stride
        box_data = cls_data = None

        for data, shape in output_items:
            if len(shape) != 4:
                continue
            a, b, c = shape[1], shape[2], shape[3]

            if a == grid_size and b == grid_size and (c == 4 or c == 80):  # NHWC
                if c == 4:
                    box_data = data.reshape(-1, 4)
                else:
                    cls_data = data.reshape(-1, 80)
            elif b == grid_size and c == grid_size and (a == 4 or a == 80):  # NCHW
                d = np.transpose(data, (0, 2, 3, 1))
                if a == 4:
                    box_data = d.reshape(-1, 4)
                else:
                    cls_data = d.reshape(-1, 80)

        if box_data is None or cls_data is None:
            continue

        max_scores = cls_data.max(axis=1)
        valid = max_scores >= conf_raw
        if not np.any(valid):
            continue

        v_box = box_data[valid]
        v_score = 1 / (1 + np.exp(-max_scores[valid]))
        v_id = cls_data[valid].argmax(axis=1)

        gv, gu = np.indices((grid_size, grid_size))
        grid = (np.stack((gu, gv), axis=-1).reshape(-1, 2)[valid] + 0.5)
        xyxy = np.hstack((grid - v_box[:, :2], grid + v_box[:, 2:])) * stride

        for box, s, cid in zip(xyxy, v_score, v_id):
            detections.append([*box, s, cid])

    final_res = []
    if detections:
        dets = np.array(detections)
        xywh = dets[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), dets[:, 4].tolist(), opt.score_thres, opt.nms_thres)
        for i in idx.flatten():
            d = dets[i]
            x1, y1, x2, y2 = (d[:4] / scale).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            final_res.append((int(d[5]), float(d[4]), x1, y1, x2, y2))

    logger.debug(f"\033[1;31mPost-process time = {(time() - t0) * 1000:.2f} ms\033[0m")

    logger.info(f"\033[1;32mDraw Results ({len(final_res)}): \033[0m")
    for cid, s, x1, y1, x2, y2 in final_res:
        name = coco_names[cid] if cid < len(coco_names) else str(cid)
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {s:.2f}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name}:{s:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(opt.img_save_path, img)
    logger.info(f"Saved to {opt.img_save_path}")


coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

if __name__ == "__main__":
    main()