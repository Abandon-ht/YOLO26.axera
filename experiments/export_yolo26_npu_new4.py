#!/usr/bin/env python3
# YOLO26 Export ONNX (No Mod, Global TopK, Single output [1,300,6], avoid expand/repeat shape-concat)
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

# fixed grids for 640
GRID_S8  = _make_grid_flat(80, 80)   # 6400x2
GRID_S16 = _make_grid_flat(40, 40)   # 1600x2
GRID_S32 = _make_grid_flat(20, 20)   # 400x2
STRIDES = [8.0, 16.0, 32.0]

def npu_detect_forward_global_topk_single(self, x):
    """
    Output:
      det: [1, TOPK, 6] float32
           [x1,y1,x2,y2,score,class_id_float]
    Notes:
      - no Mod/Div
      - no expand/repeat for gather indices (avoid shape Concat dtype issue)
      - assumes batch=1 (dynamic=False export)
    """
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

        # box: [1,4,H,W] -> [1,H,W,4]
        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()
        # cls: [1,80,H,W] -> [1,H,W,80]
        cls_logits = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        # best class per cell
        max_logit = torch.amax(cls_logits, dim=-1, keepdim=True)               # [1,H,W,1]
        score = torch.sigmoid(max_logit)                                       # [1,H,W,1]
        cls_id = torch.argmax(cls_logits, dim=-1, keepdim=True).to(torch.int32)  # [1,H,W,1]

        # known fixed shapes (batch=1, H/W fixed)
        _, H, W, _ = box.shape
        HW = H * W

        box_f = box.reshape(1, HW, 4)         # [1,HW,4]
        score_f = score.reshape(1, HW)        # [1,HW]
        id_f = cls_id.reshape(1, HW)          # [1,HW]

        # grid: [HW,2] -> [1,HW,2]
        grid = grid_flat.reshape(1, HW, 2) + 0.5

        # decode
        xy1 = grid - box_f[..., :2]
        xy2 = grid + box_f[..., 2:]
        xyxy = torch.cat([xy1, xy2], dim=-1) * stride   # [1,HW,4]

        xyxy_list.append(xyxy)
        score_list.append(score_f)
        id_list.append(id_f)

    # concat scales: [1,8400,4], [1,8400], [1,8400]
    xyxy_all = torch.cat(xyxy_list, dim=1)
    score_all = torch.cat(score_list, dim=1)
    id_all = torch.cat(id_list, dim=1)

    # squeeze batch -> do global topk on 1D (avoid batch-wise gather index expand)
    xyxy0 = xyxy_all.squeeze(0)    # [N,4]
    score0 = score_all.squeeze(0)  # [N]
    id0 = id_all.squeeze(0)        # [N]

    topv, topi = torch.topk(score0, k=TOPK, dim=0, largest=True, sorted=True)  # topi: [K] int64

    xyxy_topk = torch.index_select(xyxy0, dim=0, index=topi)  # [K,4]
    id_topk = torch.index_select(id0, dim=0, index=topi)      # [K]

    det0 = torch.cat([
        xyxy_topk.to(torch.float32),
        topv.unsqueeze(-1).to(torch.float32),
        id_topk.to(torch.float32).unsqueeze(-1),
    ], dim=-1)  # [K,6]

    return det0.unsqueeze(0)  # [1,K,6]


def export_npu_onnx(model_path, output_name="yolo26_global_topk_1x300x6.onnx", imgsz=640):
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    Detect.forward = npu_detect_forward_global_topk_single
    print(f"Exporting: Global TopK={TOPK}, output [1,{TOPK},6], no Mod, avoid expand/repeat...")

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
    export_npu_onnx("yolo26n.pt", "yolo26n_global_topk_1x300x6.onnx", imgsz=IMG_SIZE)