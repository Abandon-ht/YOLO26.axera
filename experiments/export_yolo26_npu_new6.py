import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import shutil

K_PER_SCALE = 200
TOPK = 300

def _make_grid_flat(H, W):
    gu, gv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    return torch.from_numpy(np.stack([gu, gv], axis=-1).reshape(-1, 2))

GRID_S8  = _make_grid_flat(80, 80)
GRID_S16 = _make_grid_flat(40, 40)
GRID_S32 = _make_grid_flat(20, 20)
STRIDES = [8.0, 16.0, 32.0]

def forward_twostage_topk(self, x):
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers, cls_layers = self.one2one_cv2, self.one2one_cv3
    else:
        box_layers, cls_layers = self.cv2, self.cv3

    grids = [GRID_S8, GRID_S16, GRID_S32]

    dets = []  # each: [1,Ki,6]

    for i in range(self.nl):
        stride = float(STRIDES[i])
        grid_flat = grids[i].to(x[i].device)

        box = box_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()
        cls = cls_layers[i](x[i]).permute(0, 2, 3, 1).contiguous()

        max_logit = torch.amax(cls, dim=-1, keepdim=True)
        score = torch.sigmoid(max_logit).reshape(1, -1)                # [1,HW]
        cid = torch.argmax(cls, dim=-1).to(torch.int32).reshape(1, -1) # [1,HW]

        _, H, W, _ = box.shape
        HW = H * W
        box_f = box.reshape(1, HW, 4)
        grid = grid_flat.reshape(1, HW, 2) + 0.5
        xyxy = torch.cat([grid - box_f[..., :2], grid + box_f[..., 2:]], dim=-1) * stride  # [1,HW,4]

        k1 = min(K_PER_SCALE, HW)
        v1, i1 = torch.topk(score, k=k1, dim=1, largest=True, sorted=True)  # [1,k1]

        x1 = torch.gather(xyxy[..., 0], 1, i1)
        y1 = torch.gather(xyxy[..., 1], 1, i1)
        x2 = torch.gather(xyxy[..., 2], 1, i1)
        y2 = torch.gather(xyxy[..., 3], 1, i1)
        xyxy_1 = torch.stack([x1, y1, x2, y2], dim=-1)  # [1,k1,4]
        cid_1 = torch.gather(cid, 1, i1).to(torch.float32).unsqueeze(-1)
        v1 = v1.to(torch.float32).unsqueeze(-1)

        dets.append(torch.cat([xyxy_1.to(torch.float32), v1, cid_1], dim=-1))  # [1,k1,6]

    det_all = torch.cat(dets, dim=1)         # [1, 3*k1, 6]
    score_all = det_all[..., 4]              # [1, 3*k1]

    v2, i2 = torch.topk(score_all, k=min(TOPK, score_all.shape[1]), dim=1, largest=True, sorted=True)

    # gather final det_all by i2 (again avoid expand: gather each column)
    cols = [torch.gather(det_all[..., j], 1, i2) for j in range(6)]
    det = torch.stack(cols, dim=-1)          # [1,TOPK,6]
    return det

def export(model_path, output_name):
    y = YOLO(model_path)
    Detect.forward = forward_twostage_topk
    p = y.export(format="onnx", imgsz=640, dynamic=False, opset=11, simplify=True)
    if p and output_name:
        shutil.move(p, output_name)
    return output_name

export("yolo26n.pt", "yolo26_fuck.onnx")