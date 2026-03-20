<h2 align="center">
  <span><img src="static/images/logo-removebg-preview (1).png" width="4%" style="transform: translate(0,9px)"></span><b>SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes</b>
</h2>

<h3 align="center">
ICLR 2026
</h3>

<div align="center" margin-bottom="6em">
<a target="_blank" rel="external nofollow noopener" href="https://xiongkunlinghu.github.io/">Xiongkun Linghu</a>,
<a target="_blank" rel="external nofollow noopener" href="https://huangjy-pku.github.io/">Jiangyong Huang</a>,
<a target="_blank" rel="external nofollow noopener" href="https://scholar.google.com/citations?user=Zhh8nbQAAAAJ&hl=en">Ziyu Zhu</a>,
<a target="_blank" rel="external nofollow noopener" href="https://buzz-beater.github.io/">Baoxiong Jia</a>,
<a target="_blank" rel="external nofollow noopener" href="https://siyuanhuang.com/">Siyuan Huang</a>
</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2510.16714" target="_blank" rel="external nofollow noopener">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://scenecot.github.io/" target="_blank" rel="external nofollow noopener">
    <img src="https://img.shields.io/badge/Page-SceneCOT-9cf" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/EricLHK/SceneCOT/tree/main" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/Data-SceneCOT185K-blue" alt="Data"></a>
    <a href="https://huggingface.co/EricLHK/SceneCOT/tree/main" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/Model-SceneCOT-pink" alt="Model"></a>
</div>
&nbsp;
<div align="middle">
<img src="static/images/teaser_v3_crop.jpg" width="85%" alt="SceneCOT Teaser">
</div>
<b>SceneCOT</b>: We propose a Chain-of-Thought reasoning method in 3D scenes (SceneCOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To our knowledge, this is the first attempt to successfully implement the COT technique for achieving human-like step-by-step reasoning for 3D scene understanding, where we show great potential in extending it to a wider range of 3D scene understanding scenarios.

### SceneCOT Framework
<div align="middle">
<img src="static/images/model_framework_v3_crop (1).jpg" width="85%" alt="LEO Teaser">
</div>
SceneCOT achieves great performance on MSQA, and Beacon3D, demonstrating the effectiveness of our reasoning framework. Especially, our method significanlty enhances the performance on counting, the most challenging task in MSQA. Our method also significanlty outperforms previous methods by a large margin in Beacon3D.



## 🔥 News
- \[2026-3\] We release training code
- \[2026-1\] SceneCOT is accepted by ICLR 2026
- \[2025-6\] We released the [webpage](https://scenecot.github.io/) of SceneCOT

## 🚀 Get Started
1. Clone Github repo.
```shell
git clone https://github.com/SceneCOT/scenecot
```

2. Create `conda` environment and install dependencies.
```shell
conda create -n scenecot python=3.9
conda activate scenecot

# install PyTorch, take our version for example
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

# install other dependencies with pip
pip install -r requirements.txt
```

3. Install third party libraries (for point cloud backbones). Installation failure may occur for `PointNext`, resulting in error when importing `PointNext`. If this happens, there are two solutions: 1) comment out the line of importing `PointNext`, or 2) download the [compiled file](https://huggingface.co/datasets/huangjy-pku/LEO_data/blob/main/pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so) and place it at `model/pointnext/cpp/pointnet2_batch/`.

```shell
# install spconv
pip install spconv-cu118

# install third-party modules
cd model

# PointNet++
cd pointnetpp
python setup.py install
cd ..

# sanity check
cd ..
PointNet++: python -c 'from model.pointnetpp.pointnetpp import PointNetPP'
```

## 📁 Prepare data and pretrained models

1. Download the dataset and the corresponding files from [data](https://huggingface.co/datasets/EricLHK/SceneCOT/tree/main).
2. Download the model checkpoints from [models](https://huggingface.co/EricLHK/SceneCOT/tree/main).

## 🕹 Training and evaluation
SceneCOT model training:
```
sh scripts/train/full_training_msqa_gqa3d.sh
```

SceneCOT model evaluation:
```
sh scripts/test/full_training_msqa_beacon3d_test_moe.sh
```

## 📊 Offline evaluation
MSQA evaluation:

You should first download the prompt and mapping file from [evaluation_assets](https://huggingface.co/datasets/EricLHK/SceneCOT/tree/main/evaluation_assets). Then set the path of `evaluation_assets` in `evaluator/msqa/configs.yaml` and other paths.

Then run: `evaluator/msqa_evaluator_offline.py`

## 📝 TODO List

- [x] Arxiv paper
- [x] Evaluation code
- [x] Training code
- [x] Model weights
- [x] SceneCOT-185K dataset


## BibTex
If you find our work helpful, please consider citing us:
```bibtex
@inproceedings{linghu2026scenecot,
  title={SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes},
  author={Linghu, Xiongkun and Huang, Jiangyong and Zhu, Ziyu and Jia, Baoxiong and Huang, Siyuan},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
