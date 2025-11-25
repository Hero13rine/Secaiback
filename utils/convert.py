#!/usr/bin/env python
"""
Convert DOTA dataset annotations so their class ids align with DIOR classes.

The class vocabularies are read from:
  - secai-common/data/dota.txt  (DOTA ordering / names)
  - secai-common/data/dior.txt  (DIOR ordering / names)

Set the four path placeholders below to the actual server directories before
running this script.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DOTA_CLASS_FILE = REPO_ROOT / "data" / "dota.txt"
DIOR_CLASS_FILE = REPO_ROOT / "data" / "dior.txt"


@dataclass
class PathConfig:
    # TODO: fill these paths on the server before running
    src_images: str = ""  # e.g., "/data/dota/images"
    src_labels: str = ""  # e.g., "/data/dota/labels"
    dst_images: str = ""  # e.g., "/data/dota_dior"
    dst_labels: str = ""  # e.g., "/data/dota_dior"


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def normalize(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def load_class_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class file not found: {path}")
    classes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def build_name_lookup(names: Iterable[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, name in enumerate(names):
        key = normalize(name)
        if key in lookup:
            raise ValueError(f"Duplicate normalized class name '{name}'")
        lookup[key] = idx
    return lookup


CUSTOM_NORMALIZATION = {
    "plane": "airplane",
    "largevehicle": "vehicle",
    "smallvehicle": "vehicle",
}


def map_dota_to_dior(dota_classes: List[str], dior_lookup: Dict[str, int]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, name in enumerate(dota_classes):
        norm = normalize(name)
        norm = CUSTOM_NORMALIZATION.get(norm, norm)
        dior_idx = dior_lookup.get(norm)
        if dior_idx is None:
            print(f"[WARN] Unable to map dataset class '{name}' to your_dataset. Its annotations will be skipped.")
            continue
        mapping[idx] = dior_idx
    return mapping


def ensure_paths(cfg: PathConfig):
    missing = [field for field, value in cfg.__dict__.items() if not value]
    if missing:
        raise ValueError(
            "Please set the following path placeholders before running: "
            + ", ".join(missing)
        )
    cfg.src_images = Path(cfg.src_images)
    cfg.src_labels = Path(cfg.src_labels)
    cfg.dst_images = Path(cfg.dst_images)
    cfg.dst_labels = Path(cfg.dst_labels)
    for path in [cfg.src_images, cfg.src_labels]:
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {path}")
    cfg.dst_images.mkdir(parents=True, exist_ok=True)
    cfg.dst_labels.mkdir(parents=True, exist_ok=True)


def find_source_image(label_rel: Path, cfg: PathConfig) -> Tuple[Path, Path] | Tuple[None, None]:
    base = label_rel.with_suffix("")
    for ext in IMAGE_EXTS:
        candidate = cfg.src_images / base.with_suffix(ext)
        if candidate.exists():
            rel_img = candidate.relative_to(cfg.src_images)
            return candidate, rel_img
    return None, None


def convert_label_file(
    label_path: Path,
    dst_path: Path,
    index_map: Dict[int, int],
    name_lookup: Dict[str, int],
) -> Tuple[int, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    lines_out: List[str] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            cls_token = parts[0]
            dior_idx = None
            try:
                dota_idx = int(float(cls_token))
                dior_idx = index_map.get(dota_idx)
            except ValueError:
                norm_name = normalize(cls_token)
                norm_name = CUSTOM_NORMALIZATION.get(norm_name, norm_name)
                dior_idx = name_lookup.get(norm_name)
            if dior_idx is None:
                skipped += 1
                continue
            parts[0] = str(dior_idx)
            lines_out.append(" ".join(parts))
            kept += 1
    dst_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept, skipped


def copy_image(src_img: Path, rel_img: Path, cfg: PathConfig):
    dst_img = cfg.dst_images / rel_img
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)


def convert_with_config(cfg: Dict[str, str]):
    """
    Run conversion with a dict config containing required paths.
    Required keys: src_images, src_labels, dst_images, dst_labels.
    Optional: src_classes, dst_classes to override class files.
    """
    path_cfg = PathConfig(
        src_images=cfg.get("src_images", ""),
        src_labels=cfg.get("src_labels", ""),
        dst_images=cfg.get("dst_images", ""),
        dst_labels=cfg.get("dst_labels", ""),
    )
    ensure_paths(path_cfg)

    dota_class_file = Path(cfg.get("src_classes", DOTA_CLASS_FILE))
    dior_class_file = Path(cfg.get("dst_classes", DIOR_CLASS_FILE))

    dota_classes = load_class_list(dota_class_file)
    dior_classes = load_class_list(dior_class_file)
    dior_lookup = build_name_lookup(dior_classes)
    index_map = map_dota_to_dior(dota_classes, dior_lookup)

    label_files = list(path_cfg.src_labels.rglob("*.txt"))
    if not label_files:
        raise RuntimeError(f"No label txt files found under {path_cfg.src_labels}")

    total_kept = total_skipped = 0
    for label_file in label_files:
        rel = label_file.relative_to(path_cfg.src_labels)
        dst_label = path_cfg.dst_labels / rel
        kept, skipped = convert_label_file(label_file, dst_label, index_map, dior_lookup)
        total_kept += kept
        total_skipped += skipped

        img_src, rel_img = find_source_image(rel, path_cfg)
        if img_src and rel_img:
            copy_image(img_src, rel_img, path_cfg)
        else:
            print(f"[WARN] Image not found for label {rel}")

    summary = (
        f"Conversion done. Kept {total_kept} annotations, skipped {total_skipped}. "
        f"Converted labels stored in {path_cfg.dst_labels}, images in {path_cfg.dst_images}."
    )
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert DOTA labels to DIOR ordering.")
    parser.add_argument("--src-images", required=True, help="Path to DOTA images root.")
    parser.add_argument("--src-labels", required=True, help="Path to DOTA labels root.")
    parser.add_argument("--dst-images", required=True, help="Output images root.")
    parser.add_argument("--dst-labels", required=True, help="Output labels root.")
    parser.add_argument("--src-classes", default=str(DOTA_CLASS_FILE), help="Path to source classes file")
    parser.add_argument("--dst-classes", default=str(DIOR_CLASS_FILE), help="Path to target classes file")
    args = parser.parse_args()

    cfg_dict = {
        "src_images": args.src_images,
        "src_labels": args.src_labels,
        "dst_images": args.dst_images,
        "dst_labels": args.dst_labels,
        "src_classes": args.src_classes,
        "dst_classes": args.dst_classes,
    }
    convert_with_config(cfg_dict)


if __name__ == "__main__":
    main()
