# Shedding Light on Images: Multi-Level Image Brightness Enhancement Guided by Arbitrary References

This is a Tensorflow implement.

### [Paper](https://www.sciencedirect.com/science/article/pii/S003132032200348X)

### Requirements ###
1. Python
2. Tensorflow >= 1.14.0
3. numpy, PIL

### Usage ###

1. Test
First download the pre-trained checkpoints, and then just run

```shell
python eval.py 
```

2. Train
First, download training data the LOL and the MIT-Adobe 5K datasets. Then, just run

```shell
python train_lol.py
python train_5k.py
```

### Citation ###
```
@article{wang2022shedding,
  title={Shedding Light on Images: Multi-Level Image Brightness Enhancement Guided by Arbitrary References},
  author={Wang, Ya’nan and Jiang, Zhuqing and Liu, Chang and Li, Kai and Men, Aidong and Wang, Haiying and Chen, Xiaobo},
  journal={Pattern Recognition},
  pages={108867},
  year={2022},
  publisher={Elsevier}
}
```
