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
import os, shutil
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

TOPK = 300

def _make_grid_flat(H, W):
    gu, gv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    return torch.from_numpy(np.stack([gu, gv], axis=-1).reshape(-1, 2))  # [HW,2]

GRID_S8  = _make_grid_flat(80, 80)
GRID_S16 = _make_grid_flat(40, 40)
GRID_S32 = _make_grid_flat(20, 20)
STRIDES = [8.0, 16.0, 32.0]

def forward_global_topk_2d(self, x):
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers, cls_layers = self.one2one_cv2, self.one2one_cv3
    else:
        box_layers, cls_layers = self.cv2, self.cv3

    grids = [GRID_S8, GRID_S16, GRID_S32]

    xyxy_all, score_all, id_all = [], [], []

    for i in range(self.nl):
        stride = float(STRIDES[i])
        grid_flat = grids[i].to(x[i].device)

        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()       # [1,H,W,4]
        cls = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()       # [1,H,W,80]

        max_logit = torch.amax(cls, dim=-1, keepdim=True)                # [1,H,W,1]
        score = torch.sigmoid(max_logit).reshape(1, -1)                  # [1,HW]
        cid = torch.argmax(cls, dim=-1).to(torch.int32).reshape(1, -1)   # [1,HW]

        _, H, W, _ = box.shape
        HW = H * W
        box_f = box.reshape(1, HW, 4)

        grid = grid_flat.reshape(1, HW, 2) + 0.5
        xy1 = grid - box_f[..., :2]
        xy2 = grid + box_f[..., 2:]
        xyxy = torch.cat([xy1, xy2], dim=-1) * stride                    # [1,HW,4]

        xyxy_all.append(xyxy)
        score_all.append(score)
        id_all.append(cid)

    xyxy_all = torch.cat(xyxy_all, dim=1)   # [1,N,4]
    score_all = torch.cat(score_all, dim=1) # [1,N]
    id_all = torch.cat(id_all, dim=1)       # [1,N]

    # TopK on 2D, dim=1 (avoid 1D TopK)
    topv, topi = torch.topk(score_all, k=TOPK, dim=1, largest=True, sorted=True)  # [1,K]

    # Gather without expand: gather each coord separately -> no shape-concat
    x1 = torch.gather(xyxy_all[..., 0], 1, topi)
    y1 = torch.gather(xyxy_all[..., 1], 1, topi)
    x2 = torch.gather(xyxy_all[..., 2], 1, topi)
    y2 = torch.gather(xyxy_all[..., 3], 1, topi)
    xyxy_topk = torch.stack([x1, y1, x2, y2], dim=-1)  # [1,K,4]

    id_topk = torch.gather(id_all, 1, topi).to(torch.float32).unsqueeze(-1)  # [1,K,1]
    score_topk = topv.to(torch.float32).unsqueeze(-1)                        # [1,K,1]

    det = torch.cat([xyxy_topk.to(torch.float32), score_topk, id_topk], dim=-1)  # [1,K,6]
    return det


def export(model_path, output_name):
    y = YOLO(model_path)
    Detect.forward = forward_global_topk_2d
    p = y.export(format="onnx", imgsz=640, dynamic=False, opset=11, simplify=True)
    if p and output_name:
        shutil.move(p, output_name)
    return output_name

if __name__ == "__main__":
    export("yolo26n.pt", "yolo26n_global_topk_fix.onnx")