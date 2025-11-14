<div id="top" align="center">

### State Space Model with Dynamic Frequency Cue for Infrared Samll Target Detection

Kuanhong Cheng, Teng Ma, Yubo Wu, and Yachao Wei </br>

[![ISJ](https://img.shields.io/badge/Elsevier-2025.105850-white.svg?style=flat-square&logo=elsevier&logoSize=auto&logoColor=white&labelColor=grey&color=blue)]()
[![ISJ](https://img.shields.io/badge/Language-Python-white.svg?style=flat-square&logo=python&logoSize=auto&logoColor=white&labelColor=grey&color=b31b1b)](https://www.python.org)

<hr/>

</div>

### Datasets Prepare

- IRSTD-1K dataset is available at [IRSTD-1K](https://github.com/RuiZhang97/ISNet).
- NUAA-SIRST dataset is available at [NUAA-SIRST](https://github.com/YimianDai/sirst).
- NUDT-SIRST dataset is available at [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection).
- DenseSIRST dataset is available at [DenseSIRST](https://github.com/GrokCV/DenseSIRST).
- We also prepare the txt file for dividing dataset and three datasets, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bCbrS5B2BWyUjK2Ic0nyreu4wZ9omgpY?usp=sharing).

<hr/>

### Commands for Taining

- The epoch and bath size for training the [DFAwareNet](https://github.com/Fortuneteller6/DFAwareNet) can be found in the following commands.

```python
python train.py --base_size 256 --crop_size 256 --epochs 500 --dataset IRSTD-1K --split_method 80_20 --model DFAwareNet --deep_supervision True --train_batch_size 4 --test_batch_size 4 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUAA-SIRST --split_method 80_20 --model DFAwareNet --deep_supervision True --train_batch_size 2 --test_batch_size 2 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUDT-SIRST --split_method 80_20 --model DFAwareNet --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 500 --dataset DenseSIRST --split_method 80_20 --model DFAwareNet --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

### Commands for Testing and Visulization

- For both testing and visulization of different dataset, you just need to change the model weights and the dataset name.

```python
python test.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_DFAwareNet_03_09_2024_18_05_39_wDS --model_dir IRSTD-1K_DFAwareNet_03_09_2024_18_05_39_wDS/mIoU__DFAwareNet_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model DFAwareNet --deep_supervision True --test_batch_size 1 --mode TXT
```

```python
python visulization.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_DFAwareNet_03_09_2024_18_05_39_wDS --model_dir IRSTD-1K_DFAwareNet_03_09_2024_18_05_39_wDS/mIoU__DFAwareNet_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model DFAwareNet --deep_supervision True --test_batch_size 1 --mode TXT
```

<hr/>

### Results and Weights

|  Methods   |    Data    |  Pd   |  Fa   |  IoU  | F1_Score |  Download   |
| :--------: | :--------: | :---: | :---: | :---: | :------: | :---------: |
| DFAwareNet |  IRSTD-1K  | 90.82 | 4.40  | 70.96 |  83.00   | [Weights](https://drive.google.com/drive/folders/1afgXFOdCgFdN9j0UB1Qmo5bq59Ml4uMI?usp=sharing) |
| DFAwareNet | NUAA-SIRST | 99.08 | 1.42  | 79.65 |  88.67   | [Weights](https://drive.google.com/drive/folders/1muNHEtXmBxy-TFh7EPQHsB7O61k3sqHs?usp=sharing) |
| DFAwareNet | NUDT-SIRST | 99.06 | 0.61  | 92.22 |  95.95   | [Weights](https://drive.google.com/drive/folders/1ijbmk7h1_YCJu07dRGOiEIDrnsDHdeGv?usp=sharing) |
| DFAwareNet | DenseSIRST | 90.21 | 10.43 | 69.78 |  82.20   | [Weights](https://drive.google.com/drive/folders/1OcEYBQHQp4pFLoQcHuLQK0IBjew74WjY?usp=sharing) |

<hr/>

### Acknowledgement

The code of this paper is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks for their awesome work.

### Citation

If you find the code helpful in your resarch or work, please cite this paper as following.

```

```

If the above article has reference value for your work, our team's other IRSTD works can also serve as references. [MDCENet](https://www.sciencedirect.com/science/article/abs/pii/S1350449524003591) | [HFMNet](https://ieeexplore.ieee.org/abstract/document/10927642)
```
@article{MDCENet,
  title={Mdcenet: Multi-dimensional cross-enhanced network for infrared small target detection},
  author={Ma, Teng and Cheng, Kuanhong and Chai, Tingting and Prasad, Shitala and Zhao, Dong and Li, Junhuai and Zhou, Huixin},
  journal={Infrared Physics \& Technology},
  volume={141},
  pages={105475},
  year={2024},
  publisher={Elsevier}
}

@article{HFMNet,
  title={A Lightweight Feature Enhancement Model for Infrared Small Target Detection}, 
  author={Cheng, Kuanhong and Ma, Teng and Fei, Rong and Li, Junhuai},
  journal={IEEE Sensors Journal}, 
  year={2025},
  volume={25},
  number={9},
  pages={15224-15234}
}

@article{WaveTD,
  title={An Wavelet Steered network for efficient infrared small target detection},
  author={Ma, Teng and Cheng, Kuanhong and Chai, Tingting and Wu, Yubo and Zhou, Huixin},
  journal={Infrared Physics \& Technology},
  volume = {148},
  pages={105850},
  year={2025},
  publisher={Elsevier}
}
```

### Contact

If you have any questions, please feel free to reach me out at teng_m@yeah.net
