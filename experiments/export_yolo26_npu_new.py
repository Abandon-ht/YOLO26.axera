#!/usr/bin/env python3
# YOLO26 Export ONNX model Script
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import os
import shutil

# ==============================================================================
# 1. 定义适配 NPU 的 Forward 函数（加入 ReduceMax/ArgMax/Sigmoid）
# ==============================================================================
def npu_detect_forward(self, x):
    """
    YOLO26 Detect Head Modified for NPU.

    Output (NHWC), per scale 3 tensors:
      - box:      (B, H, W, 4) float32
      - score:    (B, H, W, 1) float32  = sigmoid(max_logit over classes)
      - class_id: (B, H, W, 1) int32    = argmax over classes

    Total: 3 scales * 3 = 9 tensors
    """
    res = []

    # 优先使用 one2one (End-to-End) 分支
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    for i in range(self.nl):
        # Box: [B, 4, H, W] -> NHWC [B, H, W, 4]
        bboxes = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # Cls logits: [B, nc, H, W] -> NHWC [B, H, W, nc]
        cls_logits = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # 在图内做 max/argmax/sigmoid，减少后处理计算与输出带宽
        # amax -> ONNX ReduceMax
        max_logit = torch.amax(cls_logits, dim=-1, keepdim=True)      # [B,H,W,1]
        score = torch.sigmoid(max_logit)                               # [B,H,W,1]
        class_id = torch.argmax(cls_logits, dim=-1, keepdim=True)      # [B,H,W,1] (int64 in torch)
        class_id = class_id.to(torch.int32)                            # ONNX Cast -> int32（更省带宽）

        res.extend([bboxes, score, class_id])

    return res

# ==============================================================================
# 2. 执行 Monkey Patch 并导出
# ==============================================================================
def export_npu_onnx(model_path, output_name="yolo26_npu.onnx", imgsz=640):
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # 强制替换 Detect 类的 forward 方法（保持你原来的写法）
    Detect.forward = npu_detect_forward
    print("Monkey patch applied (NHWC + ReduceMax/ArgMax/Sigmoid). Starting export...")

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=11,
        simplify=True
    )

    if exported_path:
        print(f"\n✅ Export success: {exported_path}")

        if output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            shutil.move(exported_path, output_name)
            print(f"Renamed to: {output_name}")
            exported_path = output_name

        return exported_path
    else:
        print("❌ Export failed.")
        return None

if __name__ == "__main__":
    MODEL_PATH = "yolo26n.pt"
    OUTPUT_NAME = "yolo26n_npu.onnx"
    export_npu_onnx(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=640)