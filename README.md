 # BaFormer
This is the official implementation for the paper: "Efficient Temporal Action Segmentation via Boundary-aware Query Voting".
[![arXiv](https://img.shields.io/badge/arXiv-2405.15995-b31b1b.svg)](https://arxiv.org/pdf/2405.15995)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blue.svg)]([https://nips.cc/virtual/2024/poster/12345](https://proceedings.neurips.cc/paper_files/paper/2024/file/42770daf4a3384b712ea9c36e9279998-Paper-Conference.pdf))


[📄 arXiv](https://arxiv.org/pdf/2405.15995) | [🧠 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/42770daf4a3384b712ea9c36e9279998-Paper-Conference.pdf)


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
Our trained [checkpoints]() are provided below. Please place them in the `experiments/` directory before running evaluation.

## 📄 Other
If you want to try the relabeling operation, run the code with the `--relabel` option. The following shows the results before and after relabeling:

Results Before and After Relabeling
| Split | Acc (%) | F1@10 | F1@25 | F1@50 | Edit |
|:------|:-------:|:-----:|:-----:|:-----:|:-----:|
| S1 | 84.49 | 81.57 | 80.65 | 73.73 | 77.54 |
| S2 | 90.49 | 85.65 | 83.73 | 80.38 | 77.41 |
| S3 | 88.26 | 86.33 | 85.37 | 81.53 | 82.71 |
| S4 | 88.60 | 86.98 | 85.50 | 81.57 | 81.21 |
| S5 | 91.27 | 86.28 | 85.29 | 80.30 | 77.83 |

Results After Relabeling

| Split | Acc (%) | F1@10 | F1@25 | F1@50 | Edit |
|:------|:-------:|:-----:|:-----:|:-----:|:-----:|
| S1 | 84.48 | 85.71 | 84.24 | 78.82 | 80.00 |
| S2 | 90.95 | 91.00 | 89.97 | 86.38 | 85.18 |
| S3 | 88.13 | 89.22 | 88.22 | 83.21 | 84.97 |
| S4 | 88.68 | 89.23 | 88.21 | 84.62 | 83.67 |
| S5 | 91.23 | 90.19 | 89.12 | 85.41 | 82.05 |


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
