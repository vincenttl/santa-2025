"""Physics-based packing/compaction baseline for Santa 2025.

The script starts from a honeycomb-like seed placement and then runs a simple
physics simulation with collisions, gravity, damping, a shrinking square box,
and periodic shakes. The approach was tuned to target competitive leaderboard
scores when run with multiple attempts per puzzle size.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

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


@dataclass
class RunConfig:
    attempts_small: int
    attempts_medium: int
    attempts_large: int
    steps_small: int
    steps_large: int
    max_n: int
    output: str
    seed_base: int


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run the physics compaction baseline.")
    parser.add_argument("--max-n", type=int, default=200, help="Highest puzzle size (inclusive) to generate.")
    parser.add_argument("--output", type=str, default="submission_physics.csv", help="Output CSV path.")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed added to the attempt index and puzzle size for reproducibility.",
    )
    parser.add_argument("--attempts-small", type=int, default=8, help="Attempts for n <= 50.")
    parser.add_argument("--attempts-medium", type=int, default=6, help="Attempts for 51 <= n <= 100.")
    parser.add_argument("--attempts-large", type=int, default=4, help="Attempts for n > 100.")
    parser.add_argument("--steps-small", type=int, default=15000, help="Simulation steps for n < 30.")
    parser.add_argument("--steps-large", type=int, default=8000, help="Simulation steps for n >= 30.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable a faster, lower-quality run (fewer attempts and steps) suitable for quick smoke tests.",
    )

    args = parser.parse_args()

    if args.fast:
        args.attempts_small = min(args.attempts_small, 2)
        args.attempts_medium = min(args.attempts_medium, 2)
        args.attempts_large = min(args.attempts_large, 1)
        args.steps_small = min(args.steps_small, 1200)
        args.steps_large = min(args.steps_large, 800)

    return RunConfig(
        attempts_small=args.attempts_small,
        attempts_medium=args.attempts_medium,
        attempts_large=args.attempts_large,
        steps_small=args.steps_small,
        steps_large=args.steps_large,
        max_n=args.max_n,
        output=args.output,
        seed_base=args.seed_base,
    )


def attempt_counts(n: int, config: RunConfig) -> int:
    if n <= 50:
        return config.attempts_small
    if n <= 100:
        return config.attempts_medium
    return config.attempts_large


def score_positions(positions: Iterable[Tuple[float, float, float]]) -> float:
    polygons = [ChristmasTree(x, y, d).polygon for x, y, d in positions]
    bounds = unary_union(polygons).bounds
    side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / float(SCALE_FACTOR)
    return side**2 / len(polygons)


def run_submission(config: RunConfig) -> None:
    rows = []
    print(
        "Starting physics compaction submission generation...",
        f"(max_n={config.max_n}, fast={'yes' if config.steps_small < 15000 or config.steps_large < 8000 else 'no'})",
    )

    for n in tqdm(range(1, config.max_n + 1)):
        best_score = float("inf")
        best_pos = None
        attempts = attempt_counts(n, config)

        for attempt in range(attempts):
            initial = generate_good_packing(n, variant=attempt)
            compactor = PhysicsCompactor(n, initial, seed=config.seed_base + attempt + n)
            pos = compactor.run(total_steps=config.steps_small if n < 30 else config.steps_large)

            score = score_positions(pos)
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
                for i, (x, y, d) in enumerate(best_pos or [])
            ]
        )

    pd.DataFrame(rows).to_csv(config.output, index=False)
    print(f"Done! {config.output} written.")


if __name__ == "__main__":
    run_submission(parse_args())
