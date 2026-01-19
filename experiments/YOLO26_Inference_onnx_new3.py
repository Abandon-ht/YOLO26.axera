#!/usr/bin/env python3
import os
import cv2
import numpy as np
from time import time
import argparse
import logging
import onnxruntime as ort

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YOLO26")


def _is_nhwc(shape):
    return len(shape) == 4 and shape[-1] == 3


def _is_nchw(shape):
    return len(shape) == 4 and shape[1] == 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26n_topk_nomod.onnx')
    parser.add_argument('--test-img', type=str, default='bus.jpg')
    parser.add_argument('--img-save-path', type=str, default='result_yolo26.jpg')
    parser.add_argument('--score-thres', type=float, default=0.25)
    parser.add_argument('--providers', type=str, default='')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1) Load
    t0 = time()
    providers = None
    if opt.providers.strip():
        providers = [p.strip() for p in opt.providers.split(",") if p.strip()]
    sess = ort.InferenceSession(opt.model_path, providers=providers)
    logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 2) Input info
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(input_meta.shape)

    if _is_nhwc(input_shape):
        layout = "NHWC"
        m_h = int(input_shape[1] or 640)
        m_w = int(input_shape[2] or 640)
    elif _is_nchw(input_shape):
        layout = "NCHW"
        m_h = int(input_shape[2] or 640)
        m_w = int(input_shape[3] or 640)
    else:
        layout = "NCHW"
        m_h, m_w = 640, 640

    # 3) Pre-process (same as your ORT float pipeline)
    t0 = time()
    img = cv2.imread(opt.test_img)
    if img is None:
        logger.error(f"Image not found or unreadable: {opt.test_img}")
        return
    orig_h, orig_w = img.shape[:2]

    scale = min(m_h / orig_h, m_w / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    input_bgr = cv2.copyMakeBorder(
        img_resized, 0, m_h - new_h, 0, m_w - new_w,
        cv2.BORDER_CONSTANT, value=(127, 127, 127)
    )

    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        input_tensor = np.transpose(input_rgb, (2, 0, 1))[None, ...]
    else:
        input_tensor = input_rgb[None, ...]
    logger.debug(f"\033[1;31mPre-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 4) Forward
    t0 = time()
    outs = sess.run(None, {input_name: input_tensor})
    logger.debug(f"\033[1;31mForward time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 5) Post-process
    t0 = time()

    # Expected 9 outputs: (xyxy, score, id) * 3 scales
    if len(outs) != 9:
        raise RuntimeError(f"Expect 9 outputs, got {len(outs)}. Please check exported model outputs.")

    xyxy_all = []
    score_all = []
    id_all = []
    for si in range(3):
        xyxy = outs[si * 3 + 0]   # [1,K,4]
        score = outs[si * 3 + 1]  # [1,K,1]
        cid = outs[si * 3 + 2]    # [1,K,1] int32/int64

        xyxy_all.append(xyxy.reshape(-1, 4).astype(np.float32))
        score_all.append(score.reshape(-1).astype(np.float32))
        id_all.append(cid.reshape(-1).astype(np.int32))

    xyxy_all = np.concatenate(xyxy_all, axis=0)
    score_all = np.concatenate(score_all, axis=0)
    id_all = np.concatenate(id_all, axis=0)

    valid = score_all >= opt.score_thres
    xyxy_all = xyxy_all[valid]
    score_all = score_all[valid]
    id_all = id_all[valid]

    final_res = []
    for box, s, cid in zip(xyxy_all, score_all, id_all):
        x1, y1, x2, y2 = (box / scale).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        final_res.append((int(cid), float(s), x1, y1, x2, y2))

    logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 6) Draw
    logger.info(f"\033[1;32mDraw Results ({len(final_res)}): \033[0m")
    for cid, s, x1, y1, x2, y2 in final_res:
        name = coco_names[cid] if cid < len(coco_names) else str(cid)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name}:{s:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(opt.img_save_path, img)
    logger.info(f"Saved to {opt.img_save_path}")


coco_names = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

if __name__ == "__main__":
    main()