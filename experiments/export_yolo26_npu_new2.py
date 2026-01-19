#!/usr/bin/env python3
# ==============================================================================
# 1. Add ReduceMax/ArgMax/Sigmoid
# ==============================================================================
# 2. Add TopK/Gather
# ==============================================================================
import os
import shutil
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

TOPK = 300  # 你可以按需求调，比如 100/300/1000


def npu_detect_forward_topk(self, x):
    """
    Output per scale (NHWC-like but after TopK):
      xyxy:    [B, K, 4] float32 (pixel coords in padded input space)
      score:   [B, K, 1] float32
      cls_id:  [B, K, 1] int32
    Total outputs: 9 tensors (3 scales * 3).
    """
    res = []

    # prefer one2one branch
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    # fixed strides assumption for 3 scales
    # 如果你的 Detect 里有 self.stride，可用它替代
    strides = [8, 16, 32]

    for i in range(self.nl):
        stride = float(strides[i])

        # box: [B,4,H,W] -> [B,H,W,4]
        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # cls logits: [B,80,H,W] -> [B,H,W,80]
        cls_logits = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # per-cell best score/id (graph ops: ReduceMax/ArgMax/Sigmoid)
        max_logit = torch.amax(cls_logits, dim=-1, keepdim=True)            # [B,H,W,1]
        score = torch.sigmoid(max_logit)                                    # [B,H,W,1]
        cls_id = torch.argmax(cls_logits, dim=-1, keepdim=True).to(torch.int32)  # [B,H,W,1]

        B, H, W, _ = box.shape
        HW = H * W

        # flatten
        score_flat = score.reshape(B, HW)            # [B,HW]
        box_flat = box.reshape(B, HW, 4)             # [B,HW,4]
        id_flat = cls_id.reshape(B, HW)              # [B,HW]

        # TopK on score (graph op: TopK)
        k = min(TOPK, HW)
        topv, topi = torch.topk(score_flat, k=k, dim=1, largest=True, sorted=True)  # topi int64 [B,K]

        # Gather boxes/ids by topi (graph ops: Gather/GatherElements)
        idx_box = topi.unsqueeze(-1).expand(-1, -1, 4)            # [B,K,4]
        box_topk = torch.gather(box_flat, dim=1, index=idx_box)   # [B,K,4]
        id_topk = torch.gather(id_flat, dim=1, index=topi)        # [B,K]

        # Decode xyxy in graph using idx -> (u,v) (no meshgrid/range needed)
        # u = idx % W, v = idx // W
        u = torch.remainder(topi, W).to(box_topk.dtype)           # [B,K]
        v = torch.floor_divide(topi, W).to(box_topk.dtype)        # [B,K]
        grid = torch.stack((u, v), dim=-1) + 0.5                  # [B,K,2]

        xy1 = grid - box_topk[..., :2]
        xy2 = grid + box_topk[..., 2:]
        xyxy = torch.cat((xy1, xy2), dim=-1) * stride             # [B,K,4] in pixels of padded input

        res.extend([
            xyxy,
            topv.unsqueeze(-1),                 # [B,K,1]
            id_topk.unsqueeze(-1).to(torch.int32)  # [B,K,1]
        ])

    return res


def export_npu_onnx(model_path, output_name="yolo26_topk.onnx", imgsz=640):
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # monkey patch
    Detect.forward = npu_detect_forward_topk
    print(f"Monkey patch applied (TopK={TOPK}, Decode in graph). Starting export...")

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

    print("❌ Export failed.")
    return None


if __name__ == "__main__":
    MODEL_PATH = "yolo26n.pt"
    OUTPUT_NAME = "yolo26n_topk.onnx"
    export_npu_onnx(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=640)