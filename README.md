 # BaFormer
This is the official implementation for the paper: "Efficient Temporal Action Segmentation via Boundary-aware Query Voting".
[![arXiv](https://img.shields.io/badge/arXiv-2405.15995-b31b1b.svg)](https://arxiv.org/pdf/2405.15995)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://nips.cc/virtual/2024/poster/12345)
[![Supplementary](https://img.shields.io/badge/Supplementary-Material-lightgrey.svg)](https://your-link.com/supplementary.pdf)

[📄 arXiv](https://arxiv.org/pdf/2405.15995) | [🧠 NeurIPS](https://nips.cc/virtual/2024/poster/12345) | [📘 Supplementary](https://your-link.com/supplementary.pdf)


## 🚀 Overview
BaFormer delivers competitive results while requiring less FLOPs and running time. Moreover, our query-based voting mechanism significantly reduces inference time required by the single-stage model.

<div align="center">
  <img src="figures/inf_acc.png" width="80%">
  <p><em>Figure 1: Accuray vs. inference time on 50Salads. The bubble size represents the FLOPs in inference. Under different backbones, BaFormer enjoys the benefit of boundary-aware query voting with less running time and improved accuracy.</em></p>
</div>

<div align="center">
  <img src="figures/framework.png" width="99%">
  <p><em>Figure 2: It predicts query classes and masks, along with boundaries from output heads. Although each layer in the Transformer decoder holds three heads, we illustrate the three heads in the last layer for simplicity.</em></p>
</div>


## 🛠️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/peiyao-w/BaFormer.git
cd BaFormer

# 2. Create environment
conda create -n baformer python=3.8 -y
conda activate baformer

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation
(1) To obtain the features and ground-truth annotations, please download the data as described in [Data Preparation](https://github.com/yabufarha/ms-tcn), and organize it according to the following structure. Then update the data path in the code to match the location of your downloaded dataset.

```bash
data/
 ├── 50salads/
 │    ├── mapping.txt/
 │    └── splits/
 │          ├── train.split1.bundle
 │          ├── train.split2.bundle
 │          ├── train.split3.bundle
 │          ├── train.split4.bundle
 │          ├── train.split5.bundle
 │          ├── test.split1.bundle
 │          ├── test.split2.bundle
 │          ├── test.split3.bundle
 │          ├── test.split4.bundle
 │          └── test.split5.bundle
 │    └── groundTruth/
 │    └── feature/
 ├── gtea/
 └── breakfast/
```
## 🧠 Training

To train **BaFormer** on the selected dataset, run the following command:

```bash
python main.py --config ./config/50salads.yaml
```

If you have multiple GPUs, we recommend using Distributed Data Parallel (DDP) for faster training:

```bash
torchrun --nproc_per_node=8 main.py --config ./config/50salads.yaml
```

## 📈 Evaluation

After training the model, you can evaluate **BaFormer** on the test set using the following command:

```bash
python Inference.py --config ./configs/50salads.yaml --checkpoint ./results/exp_name/checkpoints/model_best.pth
```

## 📂 Checkpoints and Logs
Checkpoints and logs will be automatically saved under:

```bash
./experiments/
 ├── 50salads/
 │      └── note/
 │           ├── 1/
 │               ├── checkpoint_best.pth
 │               ├── log_plain.txt
 │               ├── log.txt
 │               ├── last_checkpoint
 │               ├── config.yaml
 │               ├── config_min.yaml
 │               ├── env.yaml
 │               └── logs_epoch/
 │           ├── 2/
              ....

 ├── gtea/
 └── breakfast/
```


## Citation
```bash
@article{wang2024efficient,
  title={Efficient temporal action segmentation via boundary-aware query voting},
  author={Wang, Peiyao and Lin, Yuewei and Blasch, Erik and Ling, Haibin and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={37765--37790},
  year={2024}
}
```
