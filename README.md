## Prepare environment.
```bash
pip install ultralytics -U
```

## Download model.
```bash
yolo predict model=yolo26n.pt source=bus.jpg
```

## Export ONNX model.
```bash
python export_yolo26_npu.py
```

## ONNX Inference.
```bash
python YOLO26_Inference_onnx.py --model-path yolo26n_npu.onnx --test-img bus.jpg
```

## AXERA Inference.
```bash
python YOLO26_Inference_axera.py --model-path yolo26n_npu3.axmodel --test-img bus.jpg
```
