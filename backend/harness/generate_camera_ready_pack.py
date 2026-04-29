# backend/harness/generate_camera_ready_pack.py
# (NEW: Create a "camera-ready pack" ZIP for Overleaf/IEEE submission integration)
#
# What it does:
#   - Collects the generated IEEEtran minipaper (.tex) and all referenced artifacts:
#       * latex/*.tex
#       * tables/*.tex
#       * figures/*.png
#       * paper/*.txt
#       * key CSVs (master_metrics_*.csv, explain/*.csv, reports/*.csv)
#       * ckg_stats.txt if present
#   - Writes a ready-to-upload ZIP under <results-dir>/camera_ready/
#
# Usage:
#   python backend/harness/generate_camera_ready_pack.py --results-dir results_1000
#
# Notes:
#   - This does not attempt full LaTeX dependency parsing; it bundles the expected directories.
#   - If you want a single-root Overleaf project, use --overleaf-root to create a folder structure.
#
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from datetime import datetime
from typing import List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    ensure_dir(os.path.dirname(dst) or ".")
    shutil.copy2(src, dst)


def copy_tree_if_exists(src_dir: str, dst_dir: str, patterns: Optional[List[str]] = None) -> None:
    if not os.path.isdir(src_dir):
        return
    ensure_dir(dst_dir)
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if patterns:
                ok = any(fn.lower().endswith(p.lower()) for p in patterns)
                if not ok:
                    continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, src_dir)
            dst = os.path.join(dst_dir, rel)
            ensure_dir(os.path.dirname(dst) or ".")
            shutil.copy2(src, dst)


def make_zip(folder: str, out_zip: str) -> None:
    ensure_dir(os.path.dirname(out_zip) or ".")
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder)
                z.write(full, rel.replace("\\", "/"))
    print(f"[MCAD/PACK] {out_zip}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a camera-ready pack ZIP from MCAD results directory.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--overleaf-root", type=str, default="", help="If set, create an Overleaf-style root folder name inside the pack.")
    p.add_argument("--paper-name", type=str, default="mcad_camera_ready", help="Base name for pack folder/zip.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_base = f"{args.paper_name}_{stamp}"
    camera_dir = os.path.join(results_dir, "camera_ready")
    ensure_dir(camera_dir)

    # Work directory (will be zipped)
    work_dir = os.path.join(camera_dir, pack_base)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    ensure_dir(work_dir)

    # Optional Overleaf root
    root = work_dir
    if args.overleaf_root.strip():
        root = os.path.join(work_dir, args.overleaf_root.strip())
        ensure_dir(root)

    # 1) LaTeX sources (generated)
    copy_tree_if_exists(os.path.join(results_dir, "latex"), os.path.join(root, "latex"), patterns=[".tex", ".txt"])

    # 2) Tables and figures
    copy_tree_if_exists(os.path.join(results_dir, "tables"), os.path.join(root, "tables"), patterns=[".tex"])
    copy_tree_if_exists(os.path.join(results_dir, "figures"), os.path.join(root, "figures"), patterns=[".png", ".pdf", ".jpg", ".jpeg"])

    # 3) Paper narratives
    copy_tree_if_exists(os.path.join(results_dir, "paper"), os.path.join(root, "paper"), patterns=[".txt", ".md"])

    # 4) Key CSVs
    for fn in (
        "master_metrics_by_session.csv",
        "master_metrics_by_type.csv",
        "metrics_by_session.csv",
        "metrics_by_type.csv",
        "ckg_stats.txt",
    ):
        copy_if_exists(os.path.join(results_dir, fn), os.path.join(root, "data", fn))

    copy_tree_if_exists(os.path.join(results_dir, "explain"), os.path.join(root, "data", "explain"), patterns=[".csv", ".json", ".txt"])
    copy_tree_if_exists(os.path.join(results_dir, "reports"), os.path.join(root, "data", "reports"), patterns=[".csv", ".txt"])

    # 5) Convenience: copy minipaper to root if exists
    minipaper_en = os.path.join(results_dir, "latex", "ieee_minipaper_en.tex")
    if os.path.exists(minipaper_en):
        copy_if_exists(minipaper_en, os.path.join(root, "main.tex"))
    else:
        # fallback: results section only
        sec = os.path.join(results_dir, "latex", "ieee_results_section_en.tex")
        if os.path.exists(sec):
            copy_if_exists(sec, os.path.join(root, "main.tex"))

    # 6) README
    readme = os.path.join(root, "README_CAMERA_READY.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("=== MCAD Camera-Ready Pack ===\n")
        f.write(f"Source results dir: {results_dir}\n")
        f.write("Contents:\n")
        f.write("- main.tex (IEEEtran skeleton)\n")
        f.write("- latex/ (generated sections)\n")
        f.write("- tables/ and figures/\n")
        f.write("- paper/ (narrative paragraphs)\n")
        f.write("- data/ (CSV reports)\n\n")
        f.write("Overleaf:\n")
        f.write("1) Upload the ZIP.\n")
        f.write("2) Set main file to main.tex.\n")
        f.write("3) Ensure IEEEtran.cls is available (Overleaf has it by default).\n")
        f.write("4) If bibliography is not ready, comment out bib lines in main.tex.\n")
    print(f"[MCAD/PACK] {readme}")

    # Zip it
    out_zip = os.path.join(camera_dir, f"{pack_base}.zip")
    make_zip(work_dir, out_zip)

    print("[MCAD/PACK] Done.")


if __name__ == "__main__":
    main()
