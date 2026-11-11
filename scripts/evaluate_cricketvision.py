"""Run real MediaPipe Pose Landmarker inference on public UJ-AQA-CricketVision clips.

This is an inference/coverage benchmark, not keypoint accuracy: the released
annotations contain action phases and body-part quality ratings, not joint
coordinates. Do not interpret coverage as Bangladesh-specific accuracy.
"""
import argparse
import hashlib
import json
import statistics
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

DATASET = "dvanderhaar/uj-aqa-cricketvision"
BASE = "https://raw.githubusercontent.com/dvanderhaar/uj-aqa-cricketvision/main/CricketVision_dataset_release"
ARCHIVES = {
    "videos": f"{BASE}/Videos.zip",
    "annotations": f"{BASE}/JSON.zip",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> int:
    count = 0
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("cricketvision_inference.json"))
    args = parser.parse_args()
    if args.max_videos < 1 or args.frames_per_video < 1:
        parser.error("--max-videos and --frames-per-video must be positive")

    root = Path(__file__).resolve().parents[1]
    model_path = root / "public/models/pose_landmarker_lite.task"
    manifest = json.loads((root / "models/manifest.json").read_text())
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_sha != manifest["sha256"]:
        raise SystemExit("Pose model checksum does not match models/manifest.json")

    with tempfile.TemporaryDirectory(prefix="golicoach-cricketvision-") as temp:
        work = Path(temp)
        hashes = {}
        extracted = {}
        for name, url in ARCHIVES.items():
            archive = work / f"{name}.zip"
            hashes[name] = download(url, archive)
            extracted[name] = safe_extract(archive, work / name)

        videos = sorted(
            path for path in (work / "videos").rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise SystemExit("No supported video files found in public Videos.zip")
        selected = videos[: args.max_videos]

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        clip_rows = []
        all_visible = []
        frame_count = 0
        detected_count = 0
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for video_path in selected:
                capture = cv2.VideoCapture(str(video_path))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                if not capture.isOpened() or total_frames < 1:
                    capture.release()
                    clip_rows.append({"clip": video_path.name, "status": "unreadable"})
                    continue

                positions = sorted({
                    min(total_frames - 1, round((i + 1) * total_frames / (args.frames_per_video + 1)))
                    for i in range(args.frames_per_video)
                })
                clip_detected = 0
                clip_visibility = []
                for frame_index in positions:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                    frame_count += 1
                    poses = result.pose_landmarks or []
                    if poses:
                        detected_count += 1
                        clip_detected += 1
                        visibility = [float(p.visibility) for p in poses[0]]
                        visible_fraction = sum(v >= 0.5 for v in visibility) / len(visibility)
                        clip_visibility.append(visible_fraction)
                        all_visible.append(visible_fraction)
                capture.release()
                clip_rows.append({
                    "clip": video_path.name,
                    "duration_seconds": round(total_frames / fps, 3) if fps > 0 else None,
                    "sampled_frames": len(positions),
                    "frames_with_pose": clip_detected,
                    "pose_detection_rate": clip_detected / len(positions) if positions else 0,
                    "mean_visible_landmark_fraction_when_detected": statistics.mean(clip_visibility) if clip_visibility else 0,
                })

        if frame_count == 0:
            raise SystemExit("No decodable video frames were processed")
        report = {
            "status": "COMPLETED",
            "dataset": DATASET,
            "dataset_description": "UJ-AQA-CricketVision public cricket batting video clips",
            "dataset_source": "https://github.com/dvanderhaar/uj-aqa-cricketvision",
            "access": "Public GitHub raw archives; no account or approval required",
            "download_sha256": hashes,
            "archive_extracted_files": extracted,
            "available_video_files": len(videos),
            "available_video_annotation_archives": extracted["annotations"],
            "sampled_video_files": len(selected),
            "frames_per_video_requested": args.frames_per_video,
            "inference_model": "MediaPipe Pose Landmarker Lite",
            "model_sha256": model_sha,
            "sampled_frames_processed": frame_count,
            "frames_with_pose_detection": detected_count,
            "pose_detection_frame_rate": detected_count / frame_count,
            "mean_visible_landmark_fraction_when_detected": statistics.mean(all_visible) if all_visible else 0,
            "clips": clip_rows,
            "interpretation": (
                "Real model inference and detection/visibility coverage on public cricket batting videos. "
                "The dataset's released labels are action phases and body-part quality ratings, not "
                "human joint coordinates. These metrics are not PCK/OKS, pose-coordinate accuracy, "
                "batting quality accuracy, or Bangladesh-specific accuracy."
            ),
            "bangladesh_accuracy": "NOT EVALUATED: this international batting dataset is not a Bangladesh-representative labeled pose benchmark.",
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))
        print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
