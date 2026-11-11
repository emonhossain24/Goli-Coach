# Cricket dataset evaluation update — 11 November 2026

## Dataset selected

Primary domain-coverage dataset: **Cric-360 v2 / `sarimshahzad/Cric-360`**, real broadcast cricket ground frames. The dataset card describes 13,545 real frames (v1+v2), including a recommended held-out test split of 1,543 frames and 408 frames tagged `ban test`. Its files require a Hugging Face account and acceptance of the dataset access conditions. The real `grounds` subset is listed as Apache-2.0; underlying broadcast rights remain with their owners. Do not use the separate synthetic `games` subset without its research-only permission.

Sources:
- Dataset card: https://huggingface.co/datasets/sarimshahzad/Cric-360
- Dataset repository: https://github.com/Cogni-Qube/Cric360-Dataset

## Evaluation code

Added `scripts/evaluate_cric360.py`. After accepting access conditions on Hugging Face and installing optional dependencies, run:

```bash
pip install datasets mediapipe numpy
huggingface-cli login
python scripts/evaluate_cric360.py --split test --limit 250 --output cric360_pose_coverage.json
```

The script runs the existing pinned MediaPipe Pose Landmarker Lite model on a deterministic sample from the dataset's test split and reports pose-detection frame rate and visible-landmark coverage, including region breakdowns. This is a cricket-domain robustness/coverage check, **not joint-coordinate accuracy**.

## Human joint annotation — required for accuracy

Cric-360 does not provide human body-joint coordinates. Its ground/pitch labels and metadata cannot be treated as body-joint ground truth. For a genuine PCK/OKS result:
1. Select a stratified sample from the held-out cricket frames (include Bangladesh-tagged frames, different camera angles, lighting and occlusion).
2. Have annotators independently label visible MediaPipe body joints 11–32 in normalized image coordinates; mark occluded/out-of-frame joints as not visible rather than guessing.
3. Have a second reviewer adjudicate ambiguous joints and record the annotation tool, annotator protocol and review status.
4. Keep all frames from the same match/sequence in one split to avoid leakage. Do not tune the model on the test split.
5. Preserve the source dataset's access and attribution terms; do not republish extracted broadcast frames.

The existing `scripts/evaluate_bangladesh.py` is specifically consent-gated for locally collected Bangladesh footage. It must not be bypassed by relabelling third-party broadcast images as consented participant footage. A separate general Cric-360 PCK evaluator should be added only after an annotation schema and rights review are approved.

## Current result status

- Cric-360 model inference: **not run in this repository session**. Dataset file access requires the account holder to accept Hugging Face conditions; no dataset frames were downloaded here.
- Human joint annotations: **not created**. No annotator-reviewed ground truth was supplied, and model-generated landmarks are not independent human labels.
- Bangladesh-specific accuracy: **NOT EVALUATED**.
- No accuracy percentage or skill score is claimed.
