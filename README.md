 # BaFormer
This is the official implementation for the paper: "Efficient Temporal Action Segmentation via Boundary-aware Query Voting".

[![arXiv](https://arxiv.org/pdf/2405.15995)]

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
  
 ├── gtea/
 └── breakfast/

## Citation

@article{wang2024efficient,
  title={Efficient temporal action segmentation via boundary-aware query voting},
  author={Wang, Peiyao and Lin, Yuewei and Blasch, Erik and Ling, Haibin and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={37765--37790},
  year={2024}
}
