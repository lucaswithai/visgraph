# End-to-End Video Instance Segmentation via Spatial-Temporal Graph Neural Networks
An implementation for paper End-to-End Video Instance Segmentation via Spatial-Temporal Graph Neural Networks (ICCV 2021).
## Getting Started

### Installation

* pytorch>=1.6.0, cuda 10.1
* detectron2
* Adelaidet
* pytorch-geometric>=1.7.0
* opencv

### Data preparation

* Put the YouTube-VIS 2019 dataset under the folder ./dataset.
* Download pretrained ```CondInst``` 1x pretrained model from [here](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md)

### Training
```bash
python tools/train_net.py --config $PATH_TO_CONFIG_FILE$ MODEL.WEIGHTS $PATH_TO_CondInst_MS_R_50_1x$
```
### Inference
```bash
python tools/test_vis.py --config-file $PATH_TO_CONFIG_FILE$ --json-file $PATH_TO_VAL_JSON_FILE$ --opts MODEL.WEIGHTS $PATH_TO_CHECKPOINT$
```

## Acknowledgement

Some codes are borrowed from [```CrossVIS```](https://github.com/hustvl/CrossVIS) and [```GSDT```](https://github.com/yongxinw/GSDT), thanks for their great work!

## Citation

```BibTeX
@inproceedings{wang2021end,
  title={End-to-End Video Instance Segmentation via Spatial-Temporal Graph Neural Networks},
  author={Wang, Tao and Xu, Ning and Chen, Kean and Lin, Weiyao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10797--10806},
  year={2021}
}
```


