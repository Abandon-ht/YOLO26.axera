#!/usr/bin/env python3
# YOLO26 Inference Script (AXERARuntime) - for ONNX outputs:
#   per scale: box(HWC,4) + score(HWC,1) + class_id(HWC,1)
# total: 9 outputs (3 scales x 3)
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
    if len(shape) == 4 and shape[-1] == 3:  # NHWC
        h = int(shape[1] or 640)
        w = int(shape[2] or 640)
        return h, w, "NHWC"
    if len(shape) == 4 and shape[1] == 3:   # NCHW
        h = int(shape[2] or 640)
        w = int(shape[3] or 640)
        return h, w, "NCHW"
    return 640, 640, "NCHW"


def to_nhwc(data):
    """Support both NHWC [1,H,W,C] and NCHW [1,C,H,W]. Return NHWC."""
    if data is None or data.ndim != 4:
        return data
    # Treat as NCHW when C is small and H==W (typical YOLO heads)
    if data.shape[1] <= 80 and data.shape[2] == data.shape[3] and data.shape[1] in (1, 4, 80):
        return np.transpose(data, (0, 2, 3, 1))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', type=str, default='yolo26n_npu3.axmodel')
    ap.add_argument('--test-img', type=str, default='bus.jpg')
    ap.add_argument('--img-save-path', type=str, default='result_yolo26.jpg')
    ap.add_argument('--score-thres', type=float, default=0.25)
    ap.add_argument('--nms-thres', type=float, default=0.7)  # kept for compatibility (optional)
    ap.add_argument('--providers', type=str, default='')
    opt = ap.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1) Load model
    t0 = time()
    providers = [p.strip() for p in opt.providers.split(",") if p.strip()] or None
    sess = ort.InferenceSession(opt.model_path, providers=providers)
    logger.debug(f"\033[1;31mLoad model time = {(time() - t0) * 1000:.2f} ms\033[0m")

    # 2) Input meta
    inp = sess.get_inputs()[0]
    input_name = inp.name
    m_h, m_w, layout = infer_hw_layout(inp.shape)

    # 3) Pre-process (keep your original uint8 pipeline)
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

    if layout == "NHWC":
        input_tensor = rgb[None, ...].astype(np.uint8)
    else:
        input_tensor = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.uint8)

    logger.debug(f"\033[1;31mPre-process time = {(time() - t0) * 1000:.2f} ms\033[0m")

    # 4) Forward
    t0 = time()
    ort_outputs = sess.run(None, {input_name: input_tensor})
    out_metas = sess.get_outputs()
    logger.debug(f"\033[1;31mForward time = {(time() - t0) * 1000:.2f} ms\033[0m")

    # 5) Post-process for new outputs: box+score+id per scale
    t0 = time()
    strides = (8, 16, 32)
    detections = []

    output_items = []
    for meta, data in zip(out_metas, ort_outputs):
        shape = list(meta.shape)
        if any(s is None or isinstance(s, str) for s in shape):
            shape = list(data.shape)
        output_items.append({"name": meta.name, "data": data, "shape": shape})

    for stride in strides:
        grid_size = m_h // stride
        box_data = score_data = id_data = None

        for item in output_items:
            data = to_nhwc(item["data"])
            if data is None or data.ndim != 4:
                continue
            n, h, w, c = data.shape
            if n != 1 or h != grid_size or w != grid_size:
                continue

            if c == 4 and box_data is None:
                box_data = data.reshape(-1, 4)
                continue

            if c == 1:
                # score: float, class_id: int
                if np.issubdtype(data.dtype, np.floating) and score_data is None:
                    score_data = data.reshape(-1).astype(np.float32)
                    continue
                if np.issubdtype(data.dtype, np.integer) and id_data is None:
                    id_data = data.reshape(-1).astype(np.int32)
                    continue

                # fallback by name hint (in case dtype is not as expected)
                name = (item.get("name") or "").lower()
                if score_data is None and ("score" in name or "conf" in name):
                    score_data = data.reshape(-1).astype(np.float32)
                elif id_data is None and ("id" in name or "cls" in name or "class" in name):
                    id_data = data.reshape(-1).astype(np.int32)

        if box_data is None or score_data is None or id_data is None:
            logger.warning(f"Missing outputs for stride={stride}: "
                           f"box={box_data is not None}, score={score_data is not None}, id={id_data is not None}")
            continue

        valid = score_data >= opt.score_thres
        if not np.any(valid):
            continue

        v_box = box_data[valid]             # [M,4]
        v_score = score_data[valid]         # [M]
        v_id = id_data[valid]               # [M]

        gv, gu = np.indices((grid_size, grid_size))
        grid = (np.stack((gu, gv), axis=-1).reshape(-1, 2)[valid] + 0.5)  # [M,2]

        xyxy = np.hstack((grid - v_box[:, :2], grid + v_box[:, 2:])).astype(np.float32) * stride  # [M,4]

        # collect
        for box, s, cid in zip(xyxy, v_score, v_id):
            detections.append([box[0], box[1], box[2], box[3], float(s), int(cid)])

    # Optional NMS (disabled by default; YOLO26 said no NMS)
    final_res = []
    if detections:
        dets = np.array(detections, dtype=np.float32)

        # If you still want NMS, uncomment below:
        # xywh = dets[:, :4].copy()
        # xywh[:, 2:] -= xywh[:, :2]
        # idx = cv2.dnn.NMSBoxes(xywh.tolist(), dets[:, 4].tolist(), opt.score_thres, opt.nms_thres)
        # for i in idx.flatten():
        #     d = dets[i]
        #     x1, y1, x2, y2 = (d[:4] / scale).astype(int)
        #     x1, y1 = max(0, x1), max(0, y1)
        #     x2, y2 = min(orig_w, x2), min(orig_h, y2)
        #     final_res.append((int(d[5]), float(d[4]), x1, y1, x2, y2))

        # No NMS: output all
        for d in dets:
            x1, y1, x2, y2 = (d[:4] / scale).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            final_res.append((int(d[5]), float(d[4]), x1, y1, x2, y2))

    logger.debug(f"\033[1;31mPost-process time = {(time() - t0) * 1000:.2f} ms\033[0m")

    # 6) Draw
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