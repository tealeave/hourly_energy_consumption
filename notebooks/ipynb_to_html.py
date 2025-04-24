#!/usr/bin/env python3
import argparse
import os
import sys

import nbformat
from nbconvert import HTMLExporter

def convert(ipynb_path: str, out_dir: str):
    if not os.path.isfile(ipynb_path):
        sys.exit(f"Error: “{ipynb_path}” does not exist or isn’t a file.")
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(ipynb_path))[0]
    out_path = os.path.join(out_dir, f"{basename}.html")

    # load notebook and export to HTML
    nb = nbformat.read(ipynb_path, as_version=4)
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)

    print(f"✔ Converted → {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Convert a Jupyter .ipynb to HTML, preserving base filename."
    )
    p.add_argument(
        "--ipynb", "-i",
        required=True,
        help="path to the source .ipynb file"
    )
    p.add_argument(
        "--outdir", "-o",
        required=True,
        help="destination directory for the .html file"
    )
    args = p.parse_args()

    convert(args.ipynb, args.outdir)

if __name__ == "__main__":
    main()
