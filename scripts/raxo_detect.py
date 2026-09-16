#!/usr/bin/env python3
"""RAXO X-ray 开放词汇目标检测自动化脚本。

本文件不重新实现模型，而是依次调用仓库中已有的 MMDetection、SAM2 和
RAXO 官方脚本，完成以下在线推理流程：

    GroundingDINO 候选框 -> SAM2 mask -> RAXO 视觉原型分类 -> DCC 过滤

运行前只需修改下方“路径配置”区域。脚本会从训练集重新采样支持实例、
生成 mask 并构建 RAXO prototypes，不使用以前从服务器复制的结果。

最终结果保存为运行目录下的 ``detections.json``，格式为标准 COCO detection
结果列表。各阶段中间文件保存在运行目录的 ``work`` 中。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 当前 PythonProject 项目根目录

# ============================================================================
# 路径配置：更换数据集、模型或服务器时，只修改这一部分。
# 可以使用 ROOT / "项目内相对路径"，也可以直接使用 Path("/绝对路径")。
# ============================================================================

# X-ray 测试图片目录。
IMAGE_DIR = ROOT / "dataset/SIXray-D/images/test"

# 测试集 COCO 标注，GroundingDINO 推理和最终 mAP 评估都会使用。
COCO_GT = ROOT / "annotations/si_xray_d/test.json"

# MMDetection 根目录、GroundingDINO 配置和权重。
MMDET_ROOT = ROOT / "mmdetection"
GDINO_CONFIG = MMDET_ROOT / "configs/grounding_dino/grounding_dino_swin_t_si_xray.py"
GDINO_CHECKPOINT = (
    MMDET_ROOT / "checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
)

# 作者 RAXO 代码目录。
RAXO_ROOT = ROOT / "third_party/RAXO"

# DINOv3 仓库、ViT-B/16 权重和特征提取配置。
# 256 / 16 = 16，因此仍生成与原 DINOv2 配置相同的 16x16 patch 网格。
DINOV3_ROOT = ROOT / "third_party/dinov3"
DINOV3_CHECKPOINT = (
    DINOV3_ROOT / "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DINO_MODEL = "dinov3_vitb16"
DINO_IMAGE_SIZE = 256
DINO_PATCH_SIZE = 16

# SAM2 代码目录和 SAM2.1 Hiera-L 权重；请按服务器实际位置修改。
SAM2_ROOT = ROOT / "third_party/sam2"
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"

# 用于重新构建 RAXO prototypes 的训练集标注和图片目录。
# SUPPORT_SHOTS 表示每个类别抽取的实例数；论文默认使用 30 个样本。
SUPPORT_SOURCE_GT = ROOT / "annotations/si_xray_d/train.json"
SUPPORT_IMAGE_DIR = ROOT / "dataset/SIXray-D/images/train"
SUPPORT_SHOTS = 30
SUPPORT_SEED = 42

# 输出目录命名信息。建议使用小写英文、数字和短横线。
DATASET_NAME = "sixray-d"          # 当前目标/测试数据集
DETECTOR_NAME = "grounding-dino"  # 候选框检测器
SUPPORT_DATASET_NAME = "sixray-d"  # 构建 prototypes 的支持集来源

# 运行编号。None 表示启动时自动使用 YYYYMMDD-HHMMSS 时间戳。
# 若要继续某次运行，请改成原目录名，例如 RUN_NAME = "20260813-153000"，
# 并同时将 REUSE_INTERMEDIATE_RESULTS 设置为 True。
RUN_NAME: str | None = None

# 所有 RAXO 实验的公共根目录。
OUTPUT_ROOT = ROOT / "experiments/raxo"

# 实验配置名称不包含 NMS、DCC 等易变参数，详细参数由日志记录。
EXPERIMENT_NAME = (
    f"target-{DATASET_NAME}_"
    f"support-{SUPPORT_DATASET_NAME}-"
    f"k{SUPPORT_SHOTS}-seed{SUPPORT_SEED}"
)

# 是否复用当前运行目录 work/ 中本次流水线产生的 JSON 和 prototype 文件。
# 默认 False：重新采样支持集并完成全部阶段；需要断点续跑时改为 True。
REUSE_INTERMEDIATE_RESULTS = False

# GPU 环境中的 Python；通常保持为当前解释器即可。
PYTHON_EXECUTABLE = sys.executable

# 可选：已有 GroundingDINO COCO-result 候选框时填写，否则保持 None。
# 示例：Path("/data/results/gdino_test.bbox.json")
EXISTING_PROPOSALS: Path | None = None

# 可选：已有包含 RLE mask 的 COCO dataset JSON 时填写，否则保持 None。
# 设置后会同时跳过 GroundingDINO 和 SAM2，并优先于 EXISTING_PROPOSALS。
EXISTING_MASKED_PROPOSALS: Path | None = None

# 可选：指定 torch hub 缓存目录；DINOv3 本身从上面的本地仓库和权重加载。
TORCH_HOME: Path | None = None

# 是否显示第三方脚本的 tqdm/Hugging Face 进度条。
# 默认 False：终端和日志均不记录进度条刷新行，但保留普通日志、警告和指标。
SHOW_PROGRESS_BARS = False


# tqdm 的典型文本格式："Inference: 42%|████...| 21/50 [00:03<00:04]"。
# 使用 search 而非严格匹配行首，以兼容带描述文字或 ANSI 控制字符的进度条。
TQDM_LINE = re.compile(r"\d{1,3}%\|.*\|\s*\d+(?:/\d+)?")


def derived(path: Path, text: str) -> Path:
    """按照作者脚本的命名规则，在 JSON 文件名后追加阶段标识。"""
    return path.with_name(path.stem + text + ".json")


def run(
    name: str,
    command: list[str],
    cwd: Path,
    *,
    dry_run: bool,
    stage_log: Path,
    pipeline_log: Path,
    inputs: list[Path] | None = None,
    outputs: list[Path] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """执行阶段命令，并记录输入、预期输出以及实际输出状态。"""
    inputs = inputs or []
    outputs = outputs or []
    input_text = "".join(f"input: {path}\n" for path in inputs)
    output_text = "".join(f"expected_output: {path}\n" for path in outputs)
    header = (
        f"\n[{name}]\n"
        f"time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"cwd: {cwd}\n"
        f"{input_text}"
        f"{output_text}"
        f"command: {shlex.join(command)}\n"
    )
    print(header, end="")
    if dry_run:
        return

    # 使用环境变量优先通知第三方库不要创建进度条。后面的文本过滤用于兼容
    # 不支持这些环境变量的旧版 tqdm 或第三方封装。
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    if not SHOW_PROGRESS_BARS:
        process_env["TQDM_DISABLE"] = "1"
        process_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    stage_log.parent.mkdir(parents=True, exist_ok=True)
    with (
        stage_log.open("w", encoding="utf-8") as stage_stream,
        pipeline_log.open("a", encoding="utf-8") as pipeline_stream,
    ):
        stage_stream.write(header)
        pipeline_stream.write(header)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if not SHOW_PROGRESS_BARS and TQDM_LINE.search(line.rstrip("\r\n")):
                continue
            print(line, end="")
            stage_stream.write(line)
            pipeline_stream.write(line)
            stage_stream.flush()
            pipeline_stream.flush()

        return_code = process.wait()
        output_status = []
        for path in outputs:
            if path.is_file():
                output_status.append(
                    f"output: {path} (created, {path.stat().st_size} bytes)"
                )
            elif path.is_dir():
                output_status.append(f"output: {path} (created, directory)")
            else:
                output_status.append(f"output: {path} (missing)")
        status_text = "".join(f"{line}\n" for line in output_status)
        footer = f"{status_text}[{name}] exit_code={return_code}\n"
        stage_stream.write(footer)
        pipeline_stream.write(footer)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def stage_note(
    name: str,
    message: str,
    stage_log: Path,
    pipeline_log: Path,
    dry_run: bool,
) -> None:
    """记录阶段复用、跳过或文件复制等不需要启动子进程的状态。"""
    text = f"\n[{name}] {message}\n"
    print(text, end="")
    if not dry_run:
        stage_log.parent.mkdir(parents=True, exist_ok=True)
        stage_log.write_text(text, encoding="utf-8")
        with pipeline_log.open("a", encoding="utf-8") as stream:
            stream.write(text)


def copy_if_needed(source: Path, target: Path, force: bool, dry_run: bool) -> None:
    """把外部中间结果复制到工作目录，避免修改用户的原始文件。"""
    if target.exists() and not force:
        return
    print(f"\n[copy] {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def parser() -> argparse.ArgumentParser:
    """仅解析运行参数；模型和数据路径统一在文件开头配置。"""
    p = argparse.ArgumentParser(
        description="GroundingDINO -> SAM2 -> RAXO -> DCC -> COCO evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--nms", type=float, default=0.8, help="候选框 class-agnostic NMS IoU 阈值"
    )
    p.add_argument(
        "--dcc-threshold",
        type=float,
        default=0.15,
        help="Descriptor Consistency Criterion 阈值",
    )
    p.add_argument("--name", default="raxo", help="中间结果文件名标识")
    p.add_argument("--no-eval", action="store_true", help="跳过最终 COCO AP 评估")
    p.add_argument("--force", action="store_true", help="忽略已有中间文件并重新计算")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行或写文件")
    return p


def main() -> None:
    """依次构建视觉原型并执行 RAXO 在线检测。"""
    args = parser().parse_args()

    # 使用短变量名构造命令，实际路径统一来自文件顶部的配置区。
    image_dir = IMAGE_DIR
    coco_gt = COCO_GT
    mmdet_root = MMDET_ROOT
    gdino_config = GDINO_CONFIG
    gdino_checkpoint = GDINO_CHECKPOINT
    raxo_root = RAXO_ROOT
    dinov3_root = DINOV3_ROOT
    dinov3_checkpoint = DINOV3_CHECKPOINT
    support_source_gt = SUPPORT_SOURCE_GT
    support_image_dir = SUPPORT_IMAGE_DIR
    sam2_root = SAM2_ROOT
    sam2_checkpoint = SAM2_CHECKPOINT
    supplied_proposals = EXISTING_PROPOSALS
    supplied_masked = EXISTING_MASKED_PROPOSALS
    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        OUTPUT_ROOT
        / DATASET_NAME
        / DETECTOR_NAME
        / EXPERIMENT_NAME
        / run_name
    )
    recompute = args.force or not REUSE_INTERMEDIATE_RESULTS

    if not args.dry_run:
        if not (dinov3_root / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"DINOv3 repository not found or incomplete: {dinov3_root}"
            )
        if not dinov3_checkpoint.is_file():
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found: {dinov3_checkpoint}"
            )

    # 作者脚本会根据输入文件名自动派生下一阶段文件名。这里预先计算所有路径，
    # 既便于阶段间传递，也能通过“文件已存在”实现简单的断点续跑。
    work_dir = output_dir / "work"
    log_dir = output_dir / "logs"
    pipeline_log = log_dir / "pipeline.log"
    support_sample_log = log_dir / "01_support_sampling.log"
    support_mask_log = log_dir / "02_support_masks.log"
    prototype_log = log_dir / "03_build_prototypes.log"
    proposal_log = log_dir / "04_grounding_dino.log"
    mask_log = log_dir / "05_detection_masks.log"
    classification_log = log_dir / "06_raxo_classification.log"
    dcc_log = log_dir / "07_dcc_filter.log"
    final_result_log = log_dir / "08_final_result.log"
    evaluation_log = log_dir / "09_coco_evaluation.log"
    support_gt = work_dir / "support_set.json"
    support_gt_with_masks = derived(support_gt, "_with_masks")
    prototypes = work_dir / "prototypes.pt"
    proposal_prefix = work_dir / "gdino_proposals"
    proposals = Path(str(proposal_prefix) + ".bbox.json")
    masked = derived(proposals, "_with_masks")

    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        pipeline_log.write_text(
            "RAXO pipeline\n"
            f"started: {datetime.now().isoformat(timespec='seconds')}\n"
            f"dataset: {DATASET_NAME}\n"
            f"detector: {DETECTOR_NAME}\n"
            f"support_dataset: {SUPPORT_DATASET_NAME}\n"
            f"dino_model: {DINO_MODEL}\n"
            f"dino_image_size: {DINO_IMAGE_SIZE}\n"
            f"dino_patch_size: {DINO_PATCH_SIZE}\n"
            f"dinov3_repo: {dinov3_root}\n"
            f"dinov3_checkpoint: {dinov3_checkpoint}\n"
            f"experiment: {EXPERIMENT_NAME}\n"
            f"run: {run_name}\n"
            f"output_dir: {output_dir}\n"
            f"reuse_intermediate_results: {REUSE_INTERMEDIATE_RESULTS}\n",
            encoding="utf-8",
        )

    # ----------------------------------------------------------------------
    # 阶段 1：从训练集为每个类别重新抽取 K 个支持实例。
    # 输出 support_set.json，仅用于本次 prototype 构建。
    # ----------------------------------------------------------------------
    if recompute or not support_gt.exists():
        command = [
            PYTHON_EXECUTABLE,
            str(ROOT / "tools_custom/sample_coco_support.py"),
            "--src",
            str(support_source_gt),
            "--out",
            str(support_gt),
            "--shots",
            str(SUPPORT_SHOTS),
            "--seed",
            str(SUPPORT_SEED),
        ]
        run(
            "Support-set sampling",
            command,
            ROOT,
            dry_run=args.dry_run,
            stage_log=support_sample_log,
            pipeline_log=pipeline_log,
            inputs=[support_source_gt],
            outputs=[support_gt],
        )
    else:
        stage_note(
            "Support-set sampling",
            f"resumed from: {support_gt}",
            support_sample_log,
            pipeline_log,
            args.dry_run,
        )

    # ----------------------------------------------------------------------
    # 阶段 2：为支持实例生成 SAM2 mask，供前景/背景 prototype 聚合使用。
    # ----------------------------------------------------------------------
    if recompute or not support_gt_with_masks.exists():
        env = os.environ.copy()
        env["SAM2_CHECKPOINT"] = str(sam2_checkpoint)
        command = [
            PYTHON_EXECUTABLE,
            str(raxo_root / "raxo/obtain_masks.py"),
            "--gt",
            str(support_gt),
            "--image_path",
            str(support_image_dir),
            "--branch",
            "known",
        ]
        run(
            "Support-set SAM2 masks",
            command,
            sam2_root,
            dry_run=args.dry_run,
            stage_log=support_mask_log,
            pipeline_log=pipeline_log,
            inputs=[support_gt, support_image_dir, sam2_checkpoint],
            outputs=[support_gt_with_masks],
            env=env,
        )
    else:
        stage_note(
            "Support-set SAM2 masks",
            f"resumed from: {support_gt_with_masks}",
            support_mask_log,
            pipeline_log,
            args.dry_run,
        )

    # ----------------------------------------------------------------------
    # 阶段 3：DINOv3 根据支持实例构建类别正原型和背景负原型。
    # prototypes.pt 写入当前运行目录的 work/，不读取任何历史运行结果。
    # ----------------------------------------------------------------------
    if recompute or not prototypes.exists():
        env = os.environ.copy()
        if TORCH_HOME:
            env["TORCH_HOME"] = str(TORCH_HOME)
        command = [
            PYTHON_EXECUTABLE,
            str(raxo_root / "raxo/build_prototypes_with_masks_prop.py"),
            "--gt",
            str(support_gt_with_masks),
            "--image_path",
            str(support_image_dir),
            "--out",
            str(prototypes),
            "--dino_repo",
            str(dinov3_root),
            "--dino_weights",
            str(dinov3_checkpoint),
            "--dino_model",
            DINO_MODEL,
            "--dino_image_size",
            str(DINO_IMAGE_SIZE),
            "--dino_patch_size",
            str(DINO_PATCH_SIZE),
        ]
        run(
            "Build RAXO prototypes",
            command,
            raxo_root,
            dry_run=args.dry_run,
            stage_log=prototype_log,
            pipeline_log=pipeline_log,
            inputs=[
                support_gt_with_masks,
                support_image_dir,
                dinov3_root,
                dinov3_checkpoint,
            ],
            outputs=[prototypes],
            env=env,
        )
    else:
        stage_note(
            "Build RAXO prototypes",
            f"resumed from: {prototypes}",
            prototype_log,
            pipeline_log,
            args.dry_run,
        )

    # ----------------------------------------------------------------------
    # 阶段 4：获得开放词汇候选框。
    # 输入：测试图像、COCO 类别词表、GroundingDINO 配置与权重。
    # 输出：gdino_proposals.bbox.json（COCO detection result 格式）。
    # 若配置了已有检测 mask，则阶段 4 和阶段 5 都会直接跳过。
    # ----------------------------------------------------------------------
    if supplied_masked:
        copy_if_needed(supplied_masked, masked, recompute, args.dry_run)
        stage_note(
            "GroundingDINO proposals",
            f"skipped; source: {supplied_masked}\noutput: {masked}",
            proposal_log,
            pipeline_log,
            args.dry_run,
        )
        stage_note(
            "SAM2 masks",
            f"skipped; masks already copied to: {masked}",
            mask_log,
            pipeline_log,
            args.dry_run,
        )
    else:
        if supplied_proposals:
            copy_if_needed(supplied_proposals, proposals, recompute, args.dry_run)
            stage_note(
                "GroundingDINO proposals",
                f"skipped; source: {supplied_proposals}\noutput: {proposals}",
                proposal_log,
                pipeline_log,
                args.dry_run,
            )
        elif recompute or not proposals.exists():
            command = [
                PYTHON_EXECUTABLE,
                str(mmdet_root / "tools/test.py"),
                str(gdino_config),
                str(gdino_checkpoint),
                "--work-dir",
                str(work_dir / "grounding_dino"),
                "--cfg-options",
                f"test_dataloader.dataset.ann_file={coco_gt}",
                f"test_dataloader.dataset.data_prefix.img={image_dir}",
                f"test_evaluator.ann_file={coco_gt}",
                f"test_evaluator.outfile_prefix={proposal_prefix}",
            ]
            run(
                "GroundingDINO proposals",
                command,
                mmdet_root,
                dry_run=args.dry_run,
                stage_log=proposal_log,
                pipeline_log=pipeline_log,
                inputs=[coco_gt, image_dir, gdino_config, gdino_checkpoint],
                outputs=[proposals, work_dir / "grounding_dino"],
            )
        else:
            stage_note(
                "GroundingDINO proposals",
                f"resumed from: {proposals}",
                proposal_log,
                pipeline_log,
                args.dry_run,
            )

        # ------------------------------------------------------------------
        # 阶段 5：以每个候选框作为 box prompt，让 SAM2 生成前景 RLE mask。
        # 输出仍是 COCO dataset 格式，但每条 annotation 增加 ``mask`` 字段。
        # ------------------------------------------------------------------
        if recompute or not masked.exists():
            env = os.environ.copy()
            env["SAM2_CHECKPOINT"] = str(sam2_checkpoint)
            if TORCH_HOME:
                env["TORCH_HOME"] = str(TORCH_HOME)
            command = [
                PYTHON_EXECUTABLE,
                str(raxo_root / "raxo/obtain_masks_res_inference.py"),
                "--gt",
                str(coco_gt),
                "--res",
                str(proposals),
                "--image_path",
                str(image_dir),
            ]
            run(
                "SAM2 masks",
                command,
                sam2_root,
                dry_run=args.dry_run,
                stage_log=mask_log,
                pipeline_log=pipeline_log,
                inputs=[coco_gt, proposals, image_dir, sam2_checkpoint],
                outputs=[masked],
                env=env,
            )
        else:
            stage_note(
                "SAM2 masks",
                f"resumed from: {masked}",
                mask_log,
                pipeline_log,
                args.dry_run,
            )

    # ----------------------------------------------------------------------
    # 阶段 6：RAXO 视觉分类。
    # 作者脚本先执行 class-agnostic NMS，再用 mask 聚合 DINOv3 patch 特征；
    # 随后计算区域特征与各类别/背景 prototype 的余弦相似度。匹配背景的候选框
    # 被删除，其余候选框改写为最相似的类别，并保存 DCC 所需距离。
    # ----------------------------------------------------------------------
    classified = derived(
        masked, f"_nms_{args.nms}_our_method_{args.name}"
    )
    if recompute or not classified.exists():
        env = os.environ.copy()
        if TORCH_HOME:
            env["TORCH_HOME"] = str(TORCH_HOME)
        command = [
            PYTHON_EXECUTABLE,
            str(raxo_root / "raxo/main_with_masks_prop.py"),
            "--json_res",
            str(masked),
            "--image_path",
            str(image_dir),
            "--prototypes",
            str(prototypes),
            "--nms",
            str(args.nms),
            "--name",
            args.name,
            "--branch",
            "known",
            "--dino_repo",
            str(dinov3_root),
            "--dino_weights",
            str(dinov3_checkpoint),
            "--dino_model",
            DINO_MODEL,
            "--dino_image_size",
            str(DINO_IMAGE_SIZE),
            "--dino_patch_size",
            str(DINO_PATCH_SIZE),
        ]
        run(
            "RAXO classification",
            command,
            raxo_root,
            dry_run=args.dry_run,
            stage_log=classification_log,
            pipeline_log=pipeline_log,
            inputs=[
                masked,
                image_dir,
                prototypes,
                dinov3_root,
                dinov3_checkpoint,
            ],
            outputs=[classified],
            env=env,
        )
    else:
        stage_note(
            "RAXO classification",
            f"resumed from: {classified}",
            classification_log,
            pipeline_log,
            args.dry_run,
        )

    # ----------------------------------------------------------------------
    # 阶段 7：Descriptor Consistency Criterion（DCC）。
    # 仅保留“预测类相似度 - 其他类平均相似度”大于阈值的候选框；论文默认 0.15。
    # ``_uncertanty`` 是作者原脚本使用的文件名拼写，因此这里保持一致。
    # ----------------------------------------------------------------------
    filtered = derived(classified, "_uncertanty")
    if recompute or not filtered.exists():
        command = [
            PYTHON_EXECUTABLE,
            str(raxo_root / "raxo/uncertanty_estimation_fixed.py"),
            "--dets",
            str(classified),
            "--ths",
            str(args.dcc_threshold),
        ]
        run(
            "DCC filter",
            command,
            raxo_root,
            dry_run=args.dry_run,
            stage_log=dcc_log,
            pipeline_log=pipeline_log,
            inputs=[classified],
            outputs=[filtered],
        )
    else:
        stage_note(
            "DCC filter",
            f"resumed from: {filtered}",
            dcc_log,
            pipeline_log,
            args.dry_run,
        )

    # ----------------------------------------------------------------------
    # 阶段 8：将 DCC 输出复制为稳定、简短的最终结果文件名。
    # ----------------------------------------------------------------------
    final_result = output_dir / "detections.json"
    if not args.dry_run:
        shutil.copy2(filtered, final_result)
    stage_note(
        "Final result",
        f"source: {filtered}\noutput: {final_result}",
        final_result_log,
        pipeline_log,
        args.dry_run,
    )

    # ----------------------------------------------------------------------
    # 阶段 9：可选 COCO AP 评估。只做实际图片检测时可传入 --no-eval 跳过。
    # ----------------------------------------------------------------------
    if not args.no_eval:
        command = [
            PYTHON_EXECUTABLE,
            str(raxo_root / "raxo/cocoapi_2.py"),
            "--cocoGt",
            str(coco_gt),
            "--cocoDt",
            str(final_result),
        ]
        run(
            "COCO evaluation",
            command,
            raxo_root,
            dry_run=args.dry_run,
            stage_log=evaluation_log,
            pipeline_log=pipeline_log,
            inputs=[coco_gt, final_result],
            outputs=[evaluation_log],
        )
    else:
        stage_note(
            "COCO evaluation",
            "skipped by --no-eval",
            evaluation_log,
            pipeline_log,
            args.dry_run,
        )

    print(f"\nFinal result: {final_result}")
    if not args.dry_run:
        with pipeline_log.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\nfinished: {datetime.now().isoformat(timespec='seconds')}\n"
                f"final_result: {final_result}\n"
            )


if __name__ == "__main__":
    main()
