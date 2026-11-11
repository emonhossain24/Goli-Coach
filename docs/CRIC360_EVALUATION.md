# Public cricket dataset and real model inference — 11 November 2026

## Dataset

The gated Cric-360/Hugging Face source is no longer used. The evaluation now uses the public UJ-AQA-CricketVision release by Moodley and van der Haar (WACV 2025).

- Repository: https://github.com/dvanderhaar/uj-aqa-cricketvision
- Public video archive: https://github.com/dvanderhaar/uj-aqa-cricketvision/tree/main/CricketVision_dataset_release
- Paper: https://openaccess.thecvf.com/content/WACV2025/html/Moodley_I3D-AE-LSTM_A_2-Stream_Autoencoder_for_Action_Quality_Assessment_using_a_Newly_Created_Cricket_Batsman_Video_Dataset_WACV_2025_paper.html

The repository describes 8,540 cricket stroke clips with temporal phases and body-region action-quality ratings. Its video and annotation archives are directly downloadable from public GitHub without gated account approval. Cite the authors and respect source-video rights; do not redistribute extracted clips.

## Real inference

The new scripts/evaluate_cricketvision.py downloads the public video and annotation archives, safely extracts them, samples actual frames from real cricket videos and runs the pinned MediaPipe Pose Landmarker Lite model. It records archive/model SHA-256 hashes, video/frame counts, pose detections and visible-landmark coverage.

Run locally:

```bash
pip install -r server/requirements.lock
python scripts/download_model.py
python scripts/evaluate_cricketvision.py --max-videos 12 --frames-per-video 5 --output cricketvision_inference.json
```

GitHub Actions runs a small real-inference sample and uploads its JSON report as an artifact.

## Interpretation and Bangladesh accuracy

This is real model inference on cricket video. Detection rate and landmark visibility are coverage metrics, not PCK/OKS or coordinate accuracy. CricketVision Head, Shoulder, Hands, Hips and Feet ratings are subjective action-quality labels, not per-joint coordinate ground truth.

The selected set is not Bangladesh-representative. It cannot establish Bangladesh-specific accuracy. That requires held-out Bangladesh cricket footage with independently human-labelled body-joint coordinates, adjudicated labels and player/match-separated splits. The consent-gated Bangladesh evaluator remains available for that purpose. No Bangladesh accuracy percentage is fabricated or inferred from another country's footage.

## Result status

The inference runner and CI execution are configured. A measured result is available only after the workflow completes and its cricketvision-inference artifact is produced; no metric is claimed before that run.
