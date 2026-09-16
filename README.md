# Grounding DINO + RAXO X-ray Experiments

本仓库整理了在 SIXray-D 和 VinDr-SpineXR 上将 RAXO 接入 MMDetection Grounding DINO 的实验代码。主流程为：

```text
Grounding DINO proposals → SAM2 masks → DINOv3 visual prototypes
→ RAXO classification → DCC filtering → COCO evaluation
```

## 环境安装

建议新建 Python 3.11 虚拟环境，然后在仓库根目录执行：

```bash
bash install/groundingdino/install.sh
bash install/raxo/install.sh
```

默认使用 CUDA 12.1 的 PyTorch。可通过 `PYTHON_BIN`、`TORCH_INDEX_URL`、`PROJECT_ROOT` 等环境变量覆盖默认值。DINOv3 权重需要先接受 Meta 的许可证，再把授权 URL 传给安装脚本：

```bash
DINOV3_WEIGHTS_URL='授权下载地址' bash install/raxo/install.sh
```

## 数据布局

数据集和 COCO 标注不进入 Git。运行 SIXray-D 流水线前，请按以下结构自行准备：

```text
dataset/SIXray-D/images/{train,val,test}/
annotations/si_xray_d/{train,val,test}.json
```

VinDr-SpineXR 使用：

```text
dataset/VinDr-SpineXR/images/{train,val,test}/
annotations/VinDr-SpineXR/{train,val,test}.json
```

## 运行

路径和实验参数集中在 `scripts/raxo_detect.py` 文件顶部。检查配置后可先预览所有阶段：

```bash
python scripts/raxo_detect.py --dry-run
```

执行完整流水线：

```bash
python scripts/raxo_detect.py
```

输出保存在 `experiments/raxo/`，该目录默认不进入 Git。详细上传范围见 `docs/github_upload_manifest.md`。

## 上游项目

- [MMDetection Grounding DINO](https://github.com/open-mmlab/mmdetection)
- [RAXO](https://github.com/PAGF188/RAXO)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [DINOv3](https://github.com/facebookresearch/dinov3)

公开实验结果时请同时遵守数据集、模型权重和各上游项目的许可证，并引用对应论文。
