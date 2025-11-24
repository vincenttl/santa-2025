"""
Santa 2025 Metric and layout explorer.

For each N-tree configuration, calculate the bounding square divided by N.
Final score is the sum of the scores across all configurations.

A scaling factor is used to maintain reasonably precise floating point
calculations in the shapley (v 2.1.2) library.

The module also provides a small visual experiment that compares a previous
"spread out" layout with a new attraction-based strategy that uses vibration to
pack trees tightly. Running the file directly will save two figures under the
``figures`` directory and print the scores for both approaches.
"""

import argparse
from decimal import Decimal, getcontext
from pathlib import Path
import math
import random
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Decimal precision and scaling factor
getcontext().prec = 25
scale_factor = Decimal('1e18')

# Unscaled polygon for quick visual experiments
BASE_TREE_POLYGON = Polygon(
    [
        (0.0, 0.8),
        (0.125, 0.5),
        (0.0625, 0.5),
        (0.2, 0.25),
        (0.1, 0.25),
        (0.35, 0.0),
        (0.075, 0.0),
        (0.075, -0.2),
        (-0.075, -0.2),
        (-0.075, 0.0),
        (-0.35, 0.0),
        (-0.1, 0.25),
        (-0.2, 0.25),
        (-0.0625, 0.5),
        (-0.125, 0.5),
    ]
)


class ParticipantVisibleError(Exception):
    pass


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))


def make_visual_tree(center, angle_deg):
    """Create a lightweight tree polygon for the exploratory layout plots."""

    rotated = affinity.rotate(BASE_TREE_POLYGON, angle_deg, origin=(0, 0))
    return affinity.translate(rotated, xoff=center[0], yoff=center[1])


def layout_score(polygons):
    """Match the competition metric (bounding square area / n) for polygons."""

    bounds = unary_union(polygons).bounds
    side_length = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    return (side_length**2) / len(polygons)


def resolve_overlaps(polygons, jitter, rng):
    """Nudge overlapping polygons apart using small random vibrations."""

    adjusted = polygons
    for _ in range(5):  # keep the helper bounded to avoid runaway loops
        new_polys = []
        for poly in adjusted:
            move_x = rng.uniform(-jitter, jitter)
            move_y = rng.uniform(-jitter, jitter)
            shifted = affinity.translate(poly, xoff=move_x, yoff=move_y)
            new_polys.append(shifted)

        colliding = False
        r_tree = STRtree(new_polys)
        for i, poly in enumerate(new_polys):
            for j in r_tree.query(poly):
                if i == j:
                    continue
                if poly.intersects(new_polys[j]) and not poly.touches(new_polys[j]):
                    colliding = True
                    break
            if colliding:
                break

        adjusted = new_polys
        if not colliding:
            break

    return adjusted


def spread_out_layout(num_trees, seed=0):
    """Previous approach: jitter trees along a grid to keep them apart."""

    rng = random.Random(seed)
    grid_width = math.ceil(math.sqrt(num_trees))
    spacing = 1.6
    jitter = 0.25

    polygons = []
    for idx in range(num_trees):
        row, col = divmod(idx, grid_width)
        base_x = (col - grid_width / 2) * spacing
        base_y = (row - grid_width / 2) * spacing
        center = (
            base_x + rng.uniform(-jitter, jitter),
            base_y + rng.uniform(-jitter, jitter),
        )
        angle = rng.uniform(0, 360)
        polygons.append(make_visual_tree(center, angle))

    return polygons


def magnet_layout(num_trees, seed=0, steps=30, attraction=0.15, vibration=0.1):
    """
    New approach: treat the surface like a magnet that attracts every tree
    toward the current centroid while adding vibration so trees collide and
    settle into a packed shape.
    """

    rng = random.Random(seed)
    polygons = spread_out_layout(num_trees, seed)

    for _ in range(steps):
        union_centroid = unary_union(polygons).centroid
        updated = []
        for poly in polygons:
            center = poly.centroid
            direction_x = union_centroid.x - center.x
            direction_y = union_centroid.y - center.y
            distance = math.hypot(direction_x, direction_y) or 1.0
            norm_x = direction_x / distance
            norm_y = direction_y / distance

            move_x = norm_x * attraction + rng.uniform(-vibration, vibration)
            move_y = norm_y * attraction + rng.uniform(-vibration, vibration)

            updated.append(affinity.translate(poly, xoff=move_x, yoff=move_y))

        polygons = resolve_overlaps(updated, vibration, rng)

    return polygons


def save_layout_figure(polygons, path, title):
    """Persist an SVG visual of a layout to help compare packing strategies."""

    min_x, min_y, max_x, max_y = unary_union(polygons).bounds
    padding = 0.5
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2

    def _translate(coord):
        return (
            (coord[0] - min_x) + padding,
            (max_y - coord[1]) + padding,  # flip Y for SVG coordinates
        )

    svg_parts = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>",
        f"  <title>{title}</title>",
    ]

    for poly in polygons:
        coords = " ".join(
            f"{x:.3f},{y:.3f}" for x, y in map(_translate, poly.exterior.coords)
        )
        svg_parts.append(
            "  <polygon fill='#7fbf7f' stroke='#223322' stroke-width='0.02' "
            f"points='{coords}' />"
        )

    svg_parts.append("</svg>")
    path = Path(path)
    if path.suffix != ".svg":
        path = path.with_suffix(".svg")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def compare_layouts(num_trees=25, seed=0, output_dir=Path("figures")):
    """Run both strategies and save their figures and scores."""

    output_dir.mkdir(exist_ok=True)

    spread_polys = spread_out_layout(num_trees, seed)
    magnet_polys = magnet_layout(num_trees, seed)

    spread_score = layout_score(spread_polys)
    magnet_score = layout_score(magnet_polys)

    save_layout_figure(
        spread_polys,
        output_dir / "spread_layout.svg",
        f"Spread-out layout (score={spread_score:.3f})",
    )
    save_layout_figure(
        magnet_polys,
        output_dir / "magnet_layout.svg",
        f"Magnet layout (score={magnet_score:.3f})",
    )

    summary_path = output_dir / "layout_scores.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Layout comparison results",
                f"trees={num_trees}",
                f"seed={seed}",
                f"spread_score={spread_score:.6f}",
                f"magnet_score={magnet_score:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return spread_score, magnet_score, summary_path


def score(solution, submission, row_id_column_name: str) -> float:
    """
    For each n-tree configuration, the metric calculates the bounding square
    volume divided by n, summed across all configurations.

    This metric uses shapely v2.1.2.

    Examples
    -------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> data = [['002_0', 's-0.2', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    >>> submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    >>> solution = submission[['id']].copy()
    >>> score(solution, submission, row_id_column_name)
    0.877038143325...
    """

    import pandas as pd

    # remove the leading 's' from submissions
    data_cols = ['x', 'y', 'deg']
    submission = submission.astype(str)
    for c in data_cols:
        if not submission[c].str.startswith('s').all():
            raise ParticipantVisibleError(f'Value(s) in column {c} found without `s` prefix.')
        submission[c] = submission[c].str[1:]

    # enforce value limits
    limit = 100
    bad_x = (submission['x'].astype(float) < -limit).any() or \
            (submission['x'].astype(float) > limit).any()
    bad_y = (submission['y'].astype(float) < -limit).any() or \
            (submission['y'].astype(float) > limit).any()
    if bad_x or bad_y:
        raise ParticipantVisibleError('x and/or y values outside the bounds of -100 to 100.')

    # grouping puzzles to score
    submission['tree_count_group'] = submission['id'].str.split('_').str[0]

    total_score = Decimal('0.0')
    for group, df_group in submission.groupby('tree_count_group'):
        num_trees = len(df_group)

        # Create tree objects from the submission values
        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    raise ParticipantVisibleError(f'Overlapping trees in group {group}')

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square boulding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
        total_score += group_score

    return float(total_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare a spread-out layout with the magnet/vibration layout and "
            "save SVGs plus a text summary."
        )
    )
    parser.add_argument("--trees", type=int, default=25, help="Number of trees to place")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where figures and layout_scores.txt will be written",
    )
    args = parser.parse_args()

    before, after, summary_file = compare_layouts(
        num_trees=args.trees, seed=args.seed, output_dir=args.output_dir
    )
    print(f"Previous spread-out score: {before:.3f}")
    print(f"Attraction/vibration score: {after:.3f}")
    print(f"Saved figure outputs and scores to: {summary_file.parent.resolve()}")
