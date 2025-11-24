"""Physics-based packing/compaction baseline for Santa 2025.

The script starts from a honeycomb-like seed placement and then runs a simple
physics simulation with collisions, gravity, damping, a shrinking square box,
and periodic shakes. The approach was tuned to target competitive leaderboard
scores when run with multiple attempts per puzzle size.
"""

import argparse
import math
import random
from typing import List, Tuple

import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from santa_metric import ChristmasTree, scale_factor as SCALE_FACTOR


# Fixed tight honeycomb (community-best params as of Nov 22)
def generate_good_packing(n: int, variant: int = 0) -> List[Tuple[float, float, float]]:
    """Generate a honeycomb-like starting layout with slight random jitter."""
    configs = [
        (0.718, 0.785, 0.36),  # tightest known so far
        (0.722, 0.780, 0.38),
        (0.715, 0.790, 0.35),
    ]
    col_width, row_height, stagger = configs[variant % len(configs)]
    global_flip = random.choice([True, False])

    positions: List[Tuple[float, float, float]] = []
    cols = math.ceil(math.sqrt(n * row_height / col_width))
    rows = math.ceil(n / cols)

    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * col_width + (row % 2) * stagger - (cols * col_width / 2)
        y = row * row_height - (rows * row_height / 2)
        deg = 0.0 if row % 2 == 0 else 180.0
        if global_flip:
            deg += 180
        # tiny jitter for diversity
        x += random.uniform(-0.02, 0.02)
        y += random.uniform(-0.02, 0.02)
        positions.append((x, y, deg))

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy, d) for x, y, d in positions]
    return positions


class PhysicsCompactor:
    """Physics simulator that progressively compacts trees inside a hard box."""

    def __init__(self, n: int, initial_positions: List[Tuple[float, float, float]], seed: int = 42):
        random.seed(seed)
        self.n = n
        self.particles = []
        cx = sum(x for x, y, d in initial_positions) / n
        cy = sum(y for x, y, d in initial_positions) / n

        explode = 1.0 + random.uniform(0.15, 0.35)

        for x, y, d in initial_positions:
            particle = type("Particle", (), {})()
            particle.x = cx + explode * (x - cx)
            particle.y = cy + explode * (y - cy)
            particle.deg = (d + random.uniform(-40, 40)) % 360
            particle.vx = random.gauss(0, 0.02)
            particle.vy = random.gauss(0, 0.02)
            particle.vdeg = random.gauss(0, 8)
            self.particles.append(particle)

        self.box_side = self.current_side() * 1.1 + 1.0

    def current_side(self) -> float:
        polygons = [ChristmasTree(p.x, p.y, p.deg).polygon for p in self.particles]
        bounds = unary_union(polygons).bounds
        side_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        return side_scaled / float(SCALE_FACTOR)

    def run(self, total_steps: int = 12000) -> List[Tuple[float, float, float]]:
        gravity = 0.0012
        damping = 0.985
        push_force = 0.035
        shake_interval = 250
        shrink_patience = 0

        for step in range(total_steps):
            # 1. Gravity (positive y = down)
            for particle in self.particles:
                particle.vy += gravity

            # 2. Apply velocities
            for particle in self.particles:
                particle.x += particle.vx
                particle.y += particle.vy
                particle.deg = (particle.deg + particle.vdeg) % 360

            # 3. Friction / damping
            for particle in self.particles:
                particle.vx *= damping
                particle.vy *= damping
                particle.vdeg *= damping

            # 4. Collision resolution (centroid push + torque)
            polygons = [ChristmasTree(p.x, p.y, p.deg).polygon for p in self.particles]
            tree = STRtree(polygons)
            for i in range(self.n):
                for j in tree.query(polygons[i], predicate="intersects"):
                    if j <= i:
                        continue
                    if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                        dx = self.particles[j].x - self.particles[i].x
                        dy = self.particles[j].y - self.particles[i].y
                        dist = math.hypot(dx, dy)
                        if dist < 0.01:
                            angle = random.random() * math.pi * 2
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                            dist = 1.0
                        force = push_force / dist
                        fx = dx * force
                        fy = dy * force
                        self.particles[i].vx -= fx
                        self.particles[i].vy -= fy
                        self.particles[j].vx += fx
                        self.particles[j].vy += fy
                        self.particles[i].vdeg -= 12
                        self.particles[j].vdeg += 12

            # 5. Hard box clamp + bounce
            for particle in self.particles:
                if particle.x > self.box_side / 2:
                    particle.x = self.box_side / 2
                    particle.vx *= -0.6
                elif particle.x < -self.box_side / 2:
                    particle.x = -self.box_side / 2
                    particle.vx *= -0.6
                if particle.y > self.box_side / 2:
                    particle.y = self.box_side / 2
                    particle.vy *= -0.6
                elif particle.y < -self.box_side / 2:
                    particle.y = -self.box_side / 2
                    particle.vy *= -0.6

            # 6. Progressive shrink when stable
            current_real_side = self.current_side()
            if current_real_side < self.box_side - 0.08:
                shrink_patience += 1
                if shrink_patience > 80:
                    self.box_side = max(current_real_side + 0.03, self.box_side - 0.06)
                    shrink_patience = 0

            # 7. Periodic shake to escape local minima
            if step % shake_interval == 0:
                shake_strength = 0.06 * (1 - step / total_steps)
                for particle in self.particles:
                    particle.vx += random.gauss(0, shake_strength)
                    particle.vy += random.gauss(0, shake_strength)
                    particle.vdeg += random.gauss(0, 15)

        polygons = [ChristmasTree(p.x, p.y, p.deg).polygon for p in self.particles]
        bounds = unary_union(polygons).bounds
        cx = (bounds[0] + bounds[2]) / 2 / float(SCALE_FACTOR)
        cy = (bounds[1] + bounds[3]) / 2 / float(SCALE_FACTOR)
        final_positions = [(p.x - cx, p.y - cy, p.deg % 360) for p in self.particles]
        return [(round(x, 9), round(y, 9), round(angle, 9)) for x, y, angle in final_positions]


def run_compaction(
    min_n: int,
    max_n: int,
    output_path: str,
    fast: bool,
    steps_small: int,
    steps_large: int,
) -> None:
    """Generate a submission CSV by sweeping puzzle sizes."""

    def attempt_count(size: int) -> int:
        if fast:
            return 1
        if size <= 50:
            return 8
        if size <= 100:
            return 6
        return 4

    rows = []
    print(
        f"Starting physics compaction submission generation (fast={fast}, n={min_n}..{max_n})..."
    )
    for n in tqdm(range(min_n, max_n + 1)):
        best_score = float("inf")
        best_pos = None

        for attempt in range(attempt_count(n)):
            initial = generate_good_packing(n, variant=attempt)
            compactor = PhysicsCompactor(n, initial, seed=attempt + n)
            pos = compactor.run(total_steps=steps_small if n < 30 else steps_large)

            polygons = [ChristmasTree(x, y, d).polygon for x, y, d in pos]
            bounds = unary_union(polygons).bounds
            side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / float(SCALE_FACTOR)
            score = side**2 / n
            if score < best_score:
                best_score = score
                best_pos = pos

        rows.extend(
            [
                {
                    "id": f"{n:03d}_{i}",
                    "x": f"s{x:.9f}",
                    "y": f"s{y:.9f}",
                    "deg": f"s{d:.9f}",
                }
                for i, (x, y, d) in enumerate(best_pos)
            ]
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Done! submission saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics-based compaction for Santa 2025")
    parser.add_argument("--min-n", type=int, default=1, help="Smallest puzzle size to process (inclusive)")
    parser.add_argument("--max-n", type=int, default=200, help="Largest puzzle size to process (inclusive)")
    parser.add_argument("--output", type=str, default="submission_physics.csv", help="Path to output CSV")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster (lower-quality) configuration with fewer attempts and steps",
    )
    parser.add_argument(
        "--steps-small",
        type=int,
        default=15000,
        help="Simulation steps for puzzles with n < 30 (ignored in --fast mode)",
    )
    parser.add_argument(
        "--steps-large",
        type=int,
        default=8000,
        help="Simulation steps for puzzles with n >= 30 (ignored in --fast mode)",
    )
    parser.add_argument(
        "--fast-steps-small",
        type=int,
        default=2000,
        help="Simulation steps for puzzles with n < 30 when --fast is set",
    )
    parser.add_argument(
        "--fast-steps-large",
        type=int,
        default=1200,
        help="Simulation steps for puzzles with n >= 30 when --fast is set",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    min_n = max(1, args.min_n)
    max_n = max(min_n, args.max_n)
    steps_small = args.fast_steps_small if args.fast else args.steps_small
    steps_large = args.fast_steps_large if args.fast else args.steps_large

    run_compaction(
        min_n=min_n,
        max_n=max_n,
        output_path=args.output,
        fast=bool(args.fast),
        steps_small=steps_small,
        steps_large=steps_large,
    )
