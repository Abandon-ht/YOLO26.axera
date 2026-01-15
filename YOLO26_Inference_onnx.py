#!/usr/bin/env python3
# YOLO26 Inference Script (ONNXRuntime)
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

import onnxruntime as ort

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YOLO26")

def _is_nhwc(shape):
    # shape like [1, H, W, 3] (static) or [None, H, W, 3]
    return len(shape) == 4 and shape[-1] == 3

def _is_nchw(shape):
    # shape like [1, 3, H, W]
    return len(shape) == 4 and shape[1] == 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26.onnx', help="Path to YOLO26 *.onnx Model.")
    parser.add_argument('--test-img', type=str, default='bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='result_yolo26.jpg', help='Path to Save Result Image.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--providers', type=str, default='', help='Comma separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider"')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1. Load Model (ONNXRuntime)
    t0 = time()
    providers = None
    if opt.providers.strip():
        providers = [p.strip() for p in opt.providers.split(",") if p.strip()]
    sess = ort.InferenceSession(opt.model_path, providers=providers)
    logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 2. Get Model Info
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(input_meta.shape)  # may contain None / 'N'

    # try infer H,W and layout
    # if dynamic, you must know your model input size; fallback to 640
    if _is_nhwc(input_shape):
        layout = "NHWC"
        m_h = int(input_shape[1] or 640)
        m_w = int(input_shape[2] or 640)
    elif _is_nchw(input_shape):
        layout = "NCHW"
        m_h = int(input_shape[2] or 640)
        m_w = int(input_shape[3] or 640)
    else:
        # fallback (most YOLO exports are NCHW 1x3x640x640)
        layout = "NCHW"
        m_h, m_w = 640, 640
        logger.warning(f"Unrecognized input shape {input_shape}, fallback to NCHW 640x640")

    # 3. Pre-process (BGR -> RGB, float32, normalize; keep your resize+pad strategy)
    t0 = time()
    img = cv2.imread(opt.test_img)
    if img is None:
        logger.error(f"Image not found or unreadable: {opt.test_img}")
        return

    orig_h, orig_w = img.shape[:2]
    scale = min(m_h / orig_h, m_w / orig_w)

    # Resize & Pad (Left-top align) - keep same as your original
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    input_bgr = cv2.copyMakeBorder(
        img_resized, 0, m_h - new_h, 0, m_w - new_w,
        cv2.BORDER_CONSTANT, value=(127, 127, 127)
    )

    # ONNX commonly expects RGB + float32 normalized to [0,1]
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if layout == "NCHW":
        input_tensor = np.transpose(input_rgb, (2, 0, 1))[None, ...]  # 1x3xHxW
    else:
        input_tensor = input_rgb[None, ...]  # 1xHxWx3

    logger.debug(f"\033[1;31mPre-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 4. Forward
    t0 = time()
    ort_outputs = sess.run(None, {input_name: input_tensor})
    out_metas = sess.get_outputs()
    logger.debug(f"\033[1;31mForward time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 5. Post-process (YOLO26 Optimized) - keep logic same
    t0 = time()
    strides = [8, 16, 32]
    conf_raw = -np.log(1 / opt.score_thres - 1)
    detections = []

    # Build output_items: data + shape (use ONNX output meta shape if available, else data.shape)
    output_items = []
    for meta, data in zip(out_metas, ort_outputs):
        shape = list(meta.shape)
        # replace dynamic dims with actual runtime dims
        if any(s is None or isinstance(s, str) for s in shape):
            shape = list(data.shape)
        output_items.append({'data': data, 'shape': shape})

    for stride in strides:
        grid_size = m_h // stride  # 80, 40, 20 for 640

        box_data, cls_data = None, None
        for item in output_items:
            shape = item['shape']

            # Your original assumes logical shape is (N, H, W, C) with C==4 or 80
            # Many ONNX exports are (N, C, H, W). Here we support both.
            if len(shape) != 4:
                continue

            n, a, b, c = shape[0], shape[1], shape[2], shape[3]

            # Case 1: NHWC: [1, H, W, C]
            if b == grid_size and a == grid_size and (c == 4 or c == 80):
                data = item['data']
                if c == 4:
                    box_data = data.reshape(-1, 4)
                elif c == 80:
                    cls_data = data.reshape(-1, 80)

            # Case 2: NCHW: [1, C, H, W]
            if b == grid_size and c == grid_size and (a == 4 or a == 80):
                data = item['data']
                # NCHW -> NHWC-like flatten
                data = np.transpose(data, (0, 2, 3, 1))  # 1xHxWxC
                if a == 4:
                    box_data = data.reshape(-1, 4)
                elif a == 80:
                    cls_data = data.reshape(-1, 80)

        if box_data is None or cls_data is None:
            continue

        max_scores = np.max(cls_data, axis=1)
        valid_mask = max_scores >= conf_raw
        if not np.any(valid_mask):
            continue

        v_box = box_data[valid_mask]
        v_score = 1 / (1 + np.exp(-max_scores[valid_mask]))
        v_id = np.argmax(cls_data[valid_mask], axis=1)

        gv, gu = np.indices((grid_size, grid_size))
        grid = np.stack((gu, gv), axis=-1).reshape(-1, 2)[valid_mask] + 0.5

        xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride

        for box, s, cid in zip(xyxy, v_score, v_id):
            detections.append([*box, s, cid])

    # NMS & Render
    final_res = []
    if detections:
        dets = np.array(detections)
        xywh = dets[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), dets[:, 4].tolist(), opt.score_thres, opt.nms_thres)

        for i in indices.flatten():
            d = dets[i]
            x1, y1, x2, y2 = (d[:4] / scale).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            final_res.append((int(d[5]), d[4], x1, y1, x2, y2))

    logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 6. Draw
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
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

if __name__ == "__main__":
    main()