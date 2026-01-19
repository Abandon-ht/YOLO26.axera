#!/usr/bin/env python3
import os
import shutil
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

TOPK = 300  # 可调

def _make_grid_flat(hw):  # hw = (H,W)
    H, W = hw
    gu, gv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    grid = np.stack([gu, gv], axis=-1).reshape(-1, 2)  # [HW,2]
    return torch.from_numpy(grid)  # float32

# 固定 640 输入：三个尺度 grid 常量
GRID_FLATS = [
    _make_grid_flat((80, 80)),  # stride 8
    _make_grid_flat((40, 40)),  # stride 16
    _make_grid_flat((20, 20)),  # stride 32
]
STRIDES = [8.0, 16.0, 32.0]

def npu_detect_forward_topk_no_mod(self, x):
    res = []

    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    for i in range(self.nl):
        stride = float(STRIDES[i])
        grid_flat = GRID_FLATS[i].to(x[i].device)  # [HW,2] 常量搬到设备

        # box: [B,4,H,W] -> [B,H,W,4]
        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()
        # cls logits: [B,80,H,W] -> [B,H,W,80]
        cls_logits = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # per-cell best
        max_logit = torch.amax(cls_logits, dim=-1, keepdim=True)     # [B,H,W,1]
        score = torch.sigmoid(max_logit)                             # [B,H,W,1]
        cls_id = torch.argmax(cls_logits, dim=-1, keepdim=True).to(torch.int32)  # [B,H,W,1]

        B, H, W, _ = box.shape
        HW = H * W
        k = min(TOPK, HW)

        # flatten
        score_flat = score.reshape(B, HW)      # [B,HW]
        box_flat = box.reshape(B, HW, 4)       # [B,HW,4]
        id_flat = cls_id.reshape(B, HW)        # [B,HW]

        # TopK
        topv, topi = torch.topk(score_flat, k=k, dim=1, largest=True, sorted=True)  # topi: [B,K] int64

        # Gather boxes/ids
        idx_box = topi.unsqueeze(-1).expand(-1, -1, 4)                 # [B,K,4]
        box_topk = torch.gather(box_flat, dim=1, index=idx_box)        # [B,K,4]
        id_topk = torch.gather(id_flat, dim=1, index=topi)             # [B,K]

        # 用 grid_flat + Gather 取 (u,v)，避免 Mod/Div
        # 这里假设 batch 固定为 1（dynamic=False 常见就是 1），最稳
        # grid_flat: [HW,2], topi[0]: [K]
        grid_topk = grid_flat[topi[0]].unsqueeze(0) + 0.5              # [1,K,2]

        # Decode
        xy1 = grid_topk - box_topk[..., :2]
        xy2 = grid_topk + box_topk[..., 2:]
        xyxy = torch.cat([xy1, xy2], dim=-1) * stride                  # [1,K,4]

        res.extend([
            xyxy,                       # [1,K,4]
            topv.unsqueeze(-1),         # [1,K,1]
            id_topk.unsqueeze(-1)       # [1,K,1] int32
        ])

    return res


def export_npu_onnx(model_path, output_name="yolo26_topk_nomod.onnx", imgsz=640):
    model = YOLO(model_path)
    Detect.forward = npu_detect_forward_topk_no_mod

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=11,
        simplify=True
    )

    if exported_path and output_name:
        out_dir = os.path.dirname(output_name)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        shutil.move(exported_path, output_name)
        exported_path = output_name

    print("exported:", exported_path)
    return exported_path


if __name__ == "__main__":
    export_npu_onnx("yolo26n.pt", "yolo26n_topk_nomod.onnx", 640)