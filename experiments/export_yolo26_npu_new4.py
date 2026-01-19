#!/usr/bin/env python3
# ==============================================================================
# 1. Add ReduceMax/ArgMax/Sigmoid
# ==============================================================================
# 2. Add TopK/Gather
# ==============================================================================
# 3. Remove Mod
# ==============================================================================
# 4. Concat all scales, Global TopK
# ==============================================================================
'''
AssertionError: /model.23/Concat_5_output_0 But var.dtype = <DataType.INT16: 5> vs expect_type = <DataType.INT32: 6>
'''
import os
import shutil
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

TOPK = 300
IMG_SIZE = 640

def _make_grid_flat(H, W):
    gu, gv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    grid = np.stack([gu, gv], axis=-1).reshape(-1, 2)  # [HW,2]
    return torch.from_numpy(grid)  # float32

# 固定输入 640 -> 固定网格
GRID_S8  = _make_grid_flat(80, 80)   # [6400,2]
GRID_S16 = _make_grid_flat(40, 40)   # [1600,2]
GRID_S32 = _make_grid_flat(20, 20)   # [400,2]
STRIDES = [8.0, 16.0, 32.0]

def npu_detect_forward_global_topk(self, x):
    """
    Return:
      det: [B, TOPK, 6] float32
           (x1,y1,x2,y2,score,class_id_float)
    Notes:
      - no Mod/Div needed (grid is constant + broadcast)
      - global TopK across all scales (8400 candidates -> Top300)
    """
    # 选择 one2one 分支（如果存在）
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    grid_flats = [GRID_S8, GRID_S16, GRID_S32]

    xyxy_list = []
    score_list = []
    id_list = []

    for i in range(self.nl):
        stride = float(STRIDES[i])
        grid_flat = grid_flats[i].to(x[i].device)  # [HW,2]

        # box: [B,4,H,W] -> [B,H,W,4]
        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()
        # cls logits: [B,80,H,W] -> [B,H,W,80]
        cls_logits = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # per-cell best (ReduceMax/ArgMax/Sigmoid)
        max_logit = torch.amax(cls_logits, dim=-1, keepdim=True)              # [B,H,W,1]
        score = torch.sigmoid(max_logit)                                       # [B,H,W,1]
        cls_id = torch.argmax(cls_logits, dim=-1, keepdim=True).to(torch.int32) # [B,H,W,1]

        B, H, W, _ = box.shape
        HW = H * W

        # flatten
        box_f = box.view(B, HW, 4)          # [B,HW,4]
        score_f = score.view(B, HW)         # [B,HW]
        id_f = cls_id.view(B, HW)           # [B,HW]

        # grid: [HW,2] -> [B,HW,2]
        grid = grid_flat.unsqueeze(0).expand(B, -1, -1) + 0.5                 # [B,HW,2]

        # decode xyxy (pixel coords in padded input)
        xy1 = grid - box_f[..., :2]
        xy2 = grid + box_f[..., 2:]
        xyxy = torch.cat([xy1, xy2], dim=-1) * stride                         # [B,HW,4]

        xyxy_list.append(xyxy)
        score_list.append(score_f)
        id_list.append(id_f)

    # concat all scales: N = 6400+1600+400 = 8400
    xyxy_all = torch.cat(xyxy_list, dim=1)           # [B,N,4]
    score_all = torch.cat(score_list, dim=1)         # [B,N]
    id_all = torch.cat(id_list, dim=1)               # [B,N]

    # global TopK on score
    N = score_all.shape[1]
    k = min(TOPK, N)
    topv, topi = torch.topk(score_all, k=k, dim=1, largest=True, sorted=True)  # topi: [B,K]

    # gather xyxy/id by indices
    idx_xyxy = topi.unsqueeze(-1).expand(-1, -1, 4)     # [B,K,4]
    xyxy_topk = torch.gather(xyxy_all, dim=1, index=idx_xyxy)  # [B,K,4]
    id_topk = torch.gather(id_all, dim=1, index=topi)          # [B,K]

    # merge to [B,K,6] float32 (class_id cast to float32)
    det = torch.cat([
        xyxy_topk.to(torch.float32),
        topv.unsqueeze(-1).to(torch.float32),
        id_topk.unsqueeze(-1).to(torch.float32),
    ], dim=-1)

    return det


def export_npu_onnx(model_path, output_name="yolo26_global_topk.onnx", imgsz=640):
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    Detect.forward = npu_detect_forward_global_topk
    print(f"Monkey patch applied (Global TopK={TOPK}, merged output [1,{TOPK},6], no Mod). Exporting...")

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=11,
        simplify=True
    )

    if exported_path:
        print(f"✅ Export success: {exported_path}")
        if output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.move(exported_path, output_name)
            print(f"Renamed to: {output_name}")
            return output_name
        return exported_path

    print("❌ Export failed.")
    return None


if __name__ == "__main__":
    export_npu_onnx("yolo26n.pt", "yolo26n_global_topk.onnx", imgsz=IMG_SIZE)