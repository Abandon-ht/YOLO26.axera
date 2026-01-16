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
# Compile for Axera NPU.

Get [pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/)

```bash
pulsar2 build --config yolo26_config.json
```

## AXERA Inference.
```bash
python YOLO26_Inference_axera.py --model-path yolo26n_npu3.axmodel --test-img bus.jpg
```

## Benchmark

### [LLM8850](https://docs.m5stack.com/en/ai_hardware/LLM-8850_Card) + x86

```bash
(base) m5stack@MS-7E06:~/Workspace/AXERA/axcl-samples/build$ ./examples/axcl/axcl_yolo26 -m ../yolo26n_npu3.axmodel -i ../bus.jpg
--------------------------------------
model file : ../yolo26n_npu3.axmodel
image file : ../bus.jpg
img_h, img_w : 640 640
--------------------------------------
axclrtEngineCreateContextt is done. 
axclrtEngineGetIOInfo is done. 

grpid: 0

input size: 1
    name:   images 
        1 x 640 x 640 x 3


output size: 6
    name:  output0 
        1 x 80 x 80 x 4

    name:      580 
        1 x 80 x 80 x 80

    name:      588 
        1 x 40 x 40 x 4

    name:      602 
        1 x 40 x 40 x 80

    name:      610 
        1 x 20 x 20 x 4

    name:      624 
        1 x 20 x 20 x 80

==================================================

Engine push input is done. 
--------------------------------------
post process cost time:0.43 ms 
--------------------------------------
Repeat 1 times, avg time 1.64 ms, max_time 1.64 ms, min_time 1.64 ms
--------------------------------------
detection num: 5
 0:  94%, [  51,  399,  237,  897], person
 0:  91%, [ 227,  406,  345,  861], person
 5:  91%, [  17,  233,  801,  749], bus
 0:  80%, [ 670,  389,  809,  876], person
 0:  50%, [   0,  556,   64,  878], person
--------------------------------------
```

### [LLM8850](https://docs.m5stack.com/en/ai_hardware/LLM-8850_Card) + RaspberryPi5

```bash
m5stack@raspberrypi:~/axcl-samples/build $ ./examples/axcl/axcl_yolo26 -m ../yolo26n_npu3.axmodel -i ../bus.jpg
--------------------------------------
model file : ../yolo26n_npu3.axmodel
image file : ../bus.jpg
img_h, img_w : 640 640
--------------------------------------
axclrtEngineCreateContextt is done. 
axclrtEngineGetIOInfo is done. 

grpid: 0

input size: 1
    name:   images 
        1 x 640 x 640 x 3


output size: 6
    name:  output0 
        1 x 80 x 80 x 4

    name:      580 
        1 x 80 x 80 x 80

    name:      588 
        1 x 40 x 40 x 4

    name:      602 
        1 x 40 x 40 x 80

    name:      610 
        1 x 20 x 20 x 4

    name:      624 
        1 x 20 x 20 x 80

==================================================

Engine push input is done. 
--------------------------------------
post process cost time:0.69 ms 
--------------------------------------
Repeat 1 times, avg time 1.57 ms, max_time 1.57 ms, min_time 1.57 ms
--------------------------------------
detection num: 5
 0:  94%, [  51,  399,  237,  897], person
 0:  91%, [ 227,  406,  345,  861], person
 5:  91%, [  17,  233,  801,  749], bus
 0:  80%, [ 670,  389,  809,  876], person
 0:  50%, [   0,  556,   64,  878], person
--------------------------------------
```

### AI Pyramid

```bash
root@m5stack-AI-Pyramid:~# ./install/ax650/ax_yolo26 -m yolo26n_npu3.axmodel -i bus.jpg 
--------------------------------------
model file : yolo26n_npu3.axmodel
image file : bus.jpg
img_h, img_w : 640 640
--------------------------------------
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done. 
Engine alloc io is done. 
Engine push input is done. 
--------------------------------------
post process cost time:2.75 ms 
--------------------------------------
Repeat 1 times, avg time 1.38 ms, max_time 1.38 ms, min_time 1.38 ms
--------------------------------------
detection num: 5
 0:  94%, [  51,  399,  237,  897], person
 0:  91%, [ 227,  406,  345,  861], person
 5:  91%, [  17,  233,  801,  749], bus
 0:  80%, [ 670,  389,  809,  876], person
 0:  50%, [   0,  556,   64,  878], person
--------------------------------------
```
