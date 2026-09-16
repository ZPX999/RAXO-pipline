"""Sample a deterministic K-shot support set from COCO annotations.

By default, this script reads the SIXray-D training annotations and writes
``support_<K>shot.json`` beside the source annotation file. Sampling is by
annotation (object instance), so every category receives exactly K instances.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "mmdetection" / "data" / "si_xray_d" / "train.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a K-shot COCO support set by sampling instances."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source COCO annotation file (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to support_<K>shot.json beside --src.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10,
        help="Number of object instances sampled per category (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"COCO file is missing a valid '{key}' list: {path}")
    return data


def sample_support_set(
    data: dict[str, Any], shots: int, seed: int
) -> dict[str, Any]:
    if shots <= 0:
        raise ValueError(f"--shots must be positive, got {shots}")

    annotations_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_category[annotation["category_id"]].append(annotation)

    rng = random.Random(seed)
    selected_annotations: list[dict[str, Any]] = []

    for category in data["categories"]:
        category_id = category["id"]
        candidates = annotations_by_category[category_id]
        if len(candidates) < shots:
            raise RuntimeError(
                f"Category '{category['name']}' has only {len(candidates)} "
                f"instances, fewer than the requested {shots}."
            )
        selected_annotations.extend(rng.sample(candidates, shots))

    selected_image_ids = {
        annotation["image_id"] for annotation in selected_annotations
    }
    selected_images = [
        image for image in data["images"] if image["id"] in selected_image_ids
    ]

    support = {
        key: value
        for key, value in data.items()
        if key not in {"images", "annotations"}
    }
    support["images"] = selected_images
    support["annotations"] = selected_annotations
    return support


def print_summary(support: dict[str, Any], output_path: Path) -> None:
    counts: dict[int, int] = defaultdict(int)
    for annotation in support["annotations"]:
        counts[annotation["category_id"]] += 1

    print(f"Saved: {output_path}")
    print(f"Images: {len(support['images'])}")
    print(f"Instances: {len(support['annotations'])}")
    for category in support["categories"]:
        print(f"{category['name']}: {counts[category['id']]}")


def main() -> None:
    args = parse_args()
    source_path = args.src.resolve()
    output_path = (
        args.out.resolve()
        if args.out is not None
        else source_path.with_name(f"support_{args.shots}shot.json")
    )

    data = load_coco(source_path)
    support = sample_support_set(data, shots=args.shots, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(support, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(support, output_path)


if __name__ == "__main__":
    main()
