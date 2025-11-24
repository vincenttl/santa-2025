# Santa Metric Explorer

This repository provides the metric implementation for the Santa 2025 challenge plus a small visual experiment to compare two layout strategies:

- **Spread layout**: places trees on a jittered grid to keep them apart.
- **Magnet layout**: attracts trees toward the centroid while vibrating them so collisions drive a tighter packing.

Running the module generates SVG figures for both strategies and writes their scores to a text file.

## Setup

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Generate figures and a score summary (defaults: 25 trees, seed 0, outputs to `figures/`):

```bash
python santa_metric.py
```

Customize the run by choosing the number of trees, random seed, and output directory:

```bash
python santa_metric.py --trees 50 --seed 123 --output-dir results/
```

The command writes:

- `spread_layout.svg` and `magnet_layout.svg` in the chosen directory
- `layout_scores.txt` containing the configuration and both layout scores
