[Sparse Mask R-CNN](https://arxiv.org/abs/2105.01928) implementation using PyTorch

### Install

* `pip install mmcv-full==1.5.2`
* `pip install mmdet==2.25.0`

### Train

* `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

### Results

|   Detector   | Backbone  | Neck  | LR Schedule | Box mAP | Mask mAP | Config |
|:------------:|:---------:|:-----:|:-----------:|--------:|---------:|-------:|
| Sparse R-CNN | ResNet-50 |  FPN  |     1x      |       - |        - |  exp01 |
| Sparse R-CNN | ResNet-50 |  FPN  |     1x      |       - |        - |  exp02 |
| Sparse R-CNN | ResNet-50 | PAFPN |     1x      |       - |        - |  exp03 |
| Sparse R-CNN | ResNet-50 | PAFPN |     3x      |       - |        - |  exp04 |

### TODO

* [x] [exp01](./nets/exp01.py), default [Sparse R-CNN](https://arxiv.org/abs/2105.01928)
* [x] [exp02](./nets/exp02.py), added [GN](https://arxiv.org/abs/1803.08494) and [WS](https://arxiv.org/abs/1903.10520)
* [x] [exp03](./nets/exp04.py), added [PAFPN](https://arxiv.org/abs/1803.01534)
* [x] [exp04](./nets/exp04.py), added [MOSAIC](https://arxiv.org/abs/2004.10934),
[MixUp](https://arxiv.org/abs/1710.09412) and [CopyPaste](https://arxiv.org/abs/2012.07177)

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection