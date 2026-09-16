# RAXO + Grounding DINO GitHub 上传清单

这份清单按“能复现实验、避免上传大文件且不会泄露本机信息”的原则整理。数据集和 COCO 标注不上传；`mmdetection/`、`third_party/sam2/` 和 `third_party/dinov3/` 由安装脚本下载。`third_party/RAXO/` 只保留下面列出的实验适配文件；安装脚本检测到这些文件时会直接使用，否则会下载官方 RAXO。

## 必须上传

| 文件 | 用途 |
| --- | --- |
| `README.md` | GitHub 首页、安装方法、数据布局和运行入口 |
| `scripts/raxo_detect.py` | Grounding DINO → SAM2 → RAXO → DCC → COCO 评估主入口 |
| `scripts/evaluate_groundingdino.sh` | 使用 RGB 预训练或微调权重进行 SIXray-D 测试集评估 |
| `tools_custom/sample_coco_support.py` | 按类别和随机种子抽取 few-shot 支持集 |
| `configs/grounding_dino/grounding_dino_swin_t_si_xray.py` | SIXray-D Grounding DINO 配置 |
| `configs/grounding_dino/grounding_dino_swin_t_vinr_spine.py` | VinDr-SpineXR Grounding DINO 配置 |
| `install/groundingdino/install.sh` | 固定 MMDetection 提交、安装 Grounding DINO 依赖并下载公开权重 |
| `install/raxo/install.sh` | 下载 RAXO、SAM2、DINOv3，安装依赖并下载 SAM2 权重 |
| `third_party/RAXO/README.md` | 原 RAXO 项目说明、链接与论文引用 |
| `third_party/RAXO/raxo/build_prototypes_with_masks_prop.py` | 本实验修改后的 DINOv3 prototype 构建 |
| `third_party/RAXO/raxo/main_with_masks_prop.py` | 本实验修改后的 DINOv3 prototype 分类 |
| `third_party/RAXO/raxo/obtain_masks.py` | SAM2 支持集 mask 生成 |
| `third_party/RAXO/raxo/obtain_masks_res_inference.py` | SAM2 检测结果 mask 生成 |
| `third_party/RAXO/raxo/dataset.py` | RAXO 数据读取和特征处理 |
| `third_party/RAXO/raxo/metrics.py` | RAXO 指标辅助函数 |
| `third_party/RAXO/raxo/utils2.py` | RAXO NMS 与公共工具 |
| `third_party/RAXO/raxo/uncertanty_estimation_fixed.py` | DCC/不确定性过滤 |
| `third_party/RAXO/raxo/cocoapi_2.py` | COCO 指标计算 |
| `.gitignore` | 排除权重、缓存、中间结果、环境和凭据 |
| `docs/github_upload_files.txt` | 可直接交给 `git add --pathspec-from-file` 的精确文件名单 |

RAXO 文件源自 `PAGF188/RAXO`。公开仓库时应在 README 中保留原项目链接和论文引用，并说明这些文件是本项目的实验适配版本。

## 建议上传

- `raxo_commands.md`：先把绝对路径改成环境变量后再上传。
- COCO 标注和数据集不上传；README 记录新环境需要准备的目录结构。
- 每组正式实验只保留最终指标摘要、参数、随机种子和一份成功运行日志。当前可从以下日志提取结果：
  - `mmdetection/work_dirs/SIXray-D-RGB/20260807_110114/20260807_110114.log`
  - `mmdetection/work_dirs/sixray-d-82-test/20260814_115856/20260814_115856.log`
  - `mmdetection/work_dirs/vindr_dr_gdino/20260724_173448/20260724_173448.log`
  - `experiments/raxo/.../20260814-134741/logs/09_coco_evaluation.log`
- 小型 prototype 文件可发布到 GitHub Releases；若只需复现，建议由流水线重新生成。

## 不要上传

- `commands`、`commands.md`、`third_party/RAXO/.env`：含本机路径或凭据痕迹。
- `dataset/`、`annotations/`、`.venv/`、数据集 cache、生成的文本嵌入、NLTK 下载缓存。
- `mmdetection/`、`third_party/sam2/`、`third_party/dinov3/` 的完整源码副本和它们的 `.git/`。
- 所有 `.pth`、大模型 checkpoint、`work_dirs/`、完整 `experiments/`、mask 中间 JSON 和检测结果大文件。
- `mmdetection/groundingdino_swint_ogc_mmdet-822d7e9d.pth.1` 等重复下载文件。

## 新环境安装顺序

建议使用 Python 3.11，并在同一个已激活的虚拟环境中执行：

```bash
bash install/groundingdino/install.sh
bash install/raxo/install.sh
```

默认安装 CUDA 12.1 的 PyTorch。CPU 环境可在两个命令前设置：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash install/groundingdino/install.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash install/raxo/install.sh
```

DINOv3 ViT-B/16 权重需要先接受 Meta DINOv3 许可证并取得授权 URL：

```bash
DINOV3_WEIGHTS_URL='授权下载地址' bash install/raxo/install.sh
```

## 按清单加入 Git

在初始化顶层 Git 仓库后，可执行：

```bash
git add --pathspec-from-file=docs/github_upload_files.txt
```
