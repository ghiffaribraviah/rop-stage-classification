"""
Measure clDice (+ Dice) for the vessel CHAMPION (g40_m60_fine) on Agrawal2021,
for a true head-to-head against the friend's pipeline clDice.

Reuses the champion building blocks (vessel_round7) and the same cl_dice and
loader/split used everywhere else, so the comparison is apples-to-apples.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import normalize01, segmentation_metrics  # noqa: E402
from vessel_eval import load_dataset, split_dataset  # noqa: E402
from vessel_round7 import gabor_ch, mei_ch, predict  # noqa: E402
from friend_gabor_p10 import cl_dice  # noqa: E402

W_GABOR, W_MEI, SIGMA_SET = 0.40, 0.60, "fine"


def champion_pred(d):
    soft = normalize01(W_GABOR * gabor_ch(d) + W_MEI * mei_ch(d, SIGMA_SET), d["fov"])
    return predict(soft, d["fov"])


def evaluate(subset):
    rows = []
    for d in subset:
        pred = champion_pred(d)
        m = segmentation_metrics(pred, d["gt"])
        m["cldice"] = cl_dice(pred, d["gt"])
        rows.append(m)
    keys = ["dice", "cldice", "precision", "sensitivity", "accuracy"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, len(rows)


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print("CHAMPION vessel g40_m60_fine  (clDice head-to-head)\n", flush=True)
    print("Friend claims: clDice=0.4888 Dice=0.4624", flush=True)
    print("Friend replicated (pct-clip): clDice=0.4771 Dice=0.4062\n", flush=True)
    for name, subset in [("train", train), ("test", test), ("all", data)]:
        m, n = evaluate(subset)
        print(
            f"  [{name:5s} n={n:3d}] Dice={m['dice']:.4f}  clDice={m['cldice']:.4f}  "
            f"Prec={m['precision']:.4f}  Rec={m['sensitivity']:.4f}  Acc={m['accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
