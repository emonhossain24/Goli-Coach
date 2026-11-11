"""Run a cricket-domain pose-detection coverage check on Cric-360 v2.

This is NOT joint-coordinate accuracy: Cric-360 ground frames do not ship
human body-joint ground truth. For PCK/OKS, independently annotate sampled
frames and use evaluate_bangladesh.py only for consented Bangladesh footage.
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=250)
    p.add_argument("--split", choices=["test", "validation", "train"], default="test")
    p.add_argument("--output", type=Path, default=Path("cric360_pose_coverage.json"))
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit must be positive")

    try:
        from datasets import load_dataset
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install datasets mediapipe numpy") from exc

    dataset = load_dataset("sarimshahzad/Cric-360", "grounds", split="train")
    selected = dataset.filter(lambda row: row.get("split") == args.split)
    if not selected:
        raise SystemExit(f"No rows found for split={args.split}")
    selected = selected.select(range(min(args.limit, len(selected))))
    root = Path(__file__).resolve().parents[1]
    model = root / "public/models/pose_landmarker_lite.task"
    if not model.exists():
        raise SystemExit(f"Model file not found: {model}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=5,
    )
    rows = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(selected):
            image = row["image"].convert("RGB")
            arr = np.asarray(image)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
            poses = result.pose_landmarks or []
            visible = [
                sum(1 for point in pose if point.visibility >= 0.5) / 33
                for pose in poses
            ]
            rows.append({
                "index": index,
                "version": row.get("version"),
                "region": row.get("region"),
                "source_tag": row.get("source_tag"),
                "detected_people": len(poses),
                "max_visible_landmark_fraction": max(visible, default=0.0),
                "image_width": image.width,
                "image_height": image.height,
            })

    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("region") or "unknown")].append(row)
    report = {
        "status": "COMPLETED",
        "dataset": "sarimshahzad/Cric-360",
        "dataset_split": args.split,
        "sampled_frames": len(rows),
        "frames_with_pose_detection": sum(r["detected_people"] > 0 for r in rows),
        "pose_detection_frame_rate": sum(r["detected_people"] > 0 for r in rows) / len(rows),
        "mean_max_visible_landmark_fraction": statistics.mean(r["max_visible_landmark_fraction"] for r in rows),
        "by_region": {
            key: {
                "frames": len(items),
                "frames_with_pose_detection": sum(x["detected_people"] > 0 for x in items),
                "pose_detection_frame_rate": sum(x["detected_people"] > 0 for x in items) / len(items),
            } for key, items in sorted(groups.items())
        },
        "rows": rows,
        "interpretation": (
            "Detection/visibility coverage only. This dataset does not provide "
            "human body-joint ground truth; do not call these values PCK, pose "
            "accuracy, batting skill, or coaching accuracy."
        ),
        "human_joint_annotation": "Required for coordinate accuracy (PCK/OKS); annotate independently and use a held-out split.",
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Full report: {args.output}")

if __name__ == "__main__":
    main()
