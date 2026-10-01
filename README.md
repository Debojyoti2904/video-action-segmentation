# Video Action Segmentation — Zero-Shot

A zero-shot video action segmentation pipeline that analyzes a video using **X-CLIP** and a user-defined set of action labels.

The system processes a video as a sequence of temporal clips, compares each clip with the supplied text labels using X-CLIP, and converts the resulting predictions into timestamped action segments.

No task-specific model training or fine-tuning is required.

---

## Overview

Traditional action-recognition systems are commonly trained for a fixed set of classes. This project instead uses the pretrained **X-CLIP** model for zero-shot video-text matching.

The user provides:

- A video file
- A custom list of action labels

For example:

```text
person walking
person running
person standing
```

The pipeline processes the video in temporal windows and predicts which user-provided label best matches each window. These window-level predictions are then processed into continuous temporal action segments.

---

# Repository Architecture

After organizing the sample videos, the repository is structured as follows:

```text
video-action-segmentation/
│
├── app.py
├── run.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── video_io.py
│   ├── model.py
│   ├── pipeline.py
│   ├── segment.py
│   ├── viz.py
│   └── metrics.py
│
├── evaluation/
│   ├── __init__.py
│   ├── label_keywords.json
│   └── run_eval.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_pipeline_e2e.py
│   └── test_segment.py
│
└── data/
    └── sample_videos/
        ├── vid_2.mp4
        └── vid_3.mp4
```

### Directory and File Responsibilities

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit application and interactive user interface |
| `run.py` | Command-line entry point for running video analysis |
| `src/video_io.py` | Video loading, frame sampling and temporal clip preparation |
| `src/model.py` | X-CLIP model loading and video-text inference |
| `src/pipeline.py` | Coordinates the main analysis pipeline |
| `src/segment.py` | Converts window-level predictions into temporal action segments |
| `src/viz.py` | Generates timeline visualizations |
| `src/metrics.py` | Computes evaluation metrics |
| `evaluation/run_eval.py` | Runs evaluation using predictions and ground-truth annotations |
| `evaluation/label_keywords.json` | Evaluation-related label/keyword configuration |
| `tests/test_segment.py` | Tests temporal segmentation logic |
| `tests/test_metrics.py` | Tests evaluation metrics |
| `tests/test_pipeline_e2e.py` | End-to-end pipeline test |
| `data/sample_videos/` | Local sample videos used for demonstration/testing |

> **Note:** `env/`, `__pycache__/` and `.pytest_cache/` are local/generated files and are not part of the logical project architecture.

---

# How It Works

## 1. Video Input

The pipeline starts with a video provided by the user.

The video is loaded and converted into a sequence of frames. Frames are sampled according to the configured sampling rate and prepared for temporal processing.

The frames are then divided into temporal windows:

```text
Video
  ↓
Frames
  ↓
Temporal Window 1
Temporal Window 2
Temporal Window 3
...
```

Overlapping windows can be used so that activity changes near window boundaries are not lost.

---

## 2. X-CLIP Inference

The temporal video clips are passed to the pretrained **X-CLIP** model together with the user-provided action labels.

Example:

```text
Video Clip
    │
    ▼
  X-CLIP
    ▲
    │
walking
running
standing
```

X-CLIP maps video and text representations into a shared embedding space. Similarity between the video and each text label is then used to obtain a prediction for the clip.

The labels are not restricted to a fixed training-time class list; they are supplied dynamically for each analysis.

---

## 3. Window-Level Predictions

The model produces predictions for individual temporal windows.

For example:

```text
00:00 - 00:02   walking
00:01 - 00:03   walking
00:02 - 00:04   running
00:03 - 00:05   running
```

Since windows may overlap, these predictions are not yet the final segmentation. They need to be combined and processed temporally.

---

## 4. Temporal Segmentation

The segmentation stage converts window-level predictions into continuous action intervals.

For example:

```text
Raw Predictions

walking
walking
walking
running
running
running
standing
standing
```

can be represented as:

```text
00:00 - 00:06   Walking
00:06 - 00:12   Running
00:12 - 00:16   Standing
```

The exact boundaries depend on the temporal-processing logic and configuration implemented in `src/segment.py`.

---

## 5. Visualization and Export

The detected action segments can be represented on a horizontal timeline:

```text
Time →

0s        5s        10s       15s       20s
|---------|---------|---------|---------|

[   Walking   ][ Running ][  Standing  ]
```

The segmented results can also be exported in machine-readable formats such as:

```text
segments.csv
segments.json
```

A timeline image can also be generated by the visualization module.

---

# Installation

Python **3.10+** is recommended.

## 1. Clone the repository

```bash
git clone <repository_url>
cd video-action-segmentation
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv env
env\Scripts\activate
```

### macOS / Linux

```bash
python -m venv env
source env/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Streamlit Application

Start the interactive application with:

```bash
streamlit run app.py
```

The application allows the user to upload a video and provide a custom list of action labels.

Example:

```text
person walking
person running
person standing
```

The application then runs the analysis and presents the detected temporal segments and visualization.

---

## 2. Command-Line Interface

The CLI is intended for direct execution from a terminal or automated workflows.

### Syntax

```bash
python run.py <video_path> -l "<label1>, <label2>, <label3>"
```

### Example

```bash
python run.py data/sample_videos/vid_2.mp4 -l "walking, running, standing"
```

Another example:

```bash
python run.py data/sample_videos/vid_3.mp4 -l "chopping vegetables, stirring, plating"
```

The analysis produces the configured result files, such as:

```text
segments.csv
segments.json
timeline.png
```

---

# Evaluation

The repository includes a separate evaluation workflow under `evaluation/`.

```text
Predicted Segments
        +
Ground-Truth Annotations
        ↓
evaluation/run_eval.py
        ↓
src/metrics.py
        ↓
Evaluation Metrics
```

The evaluation code can be used to compare predicted temporal segments against human-annotated ground truth.

### Frame Accuracy

Measures the proportion of evaluated frames assigned to the correct action label.

### Segment F1

Measures how well predicted action segments overlap with ground-truth segments.

### Edit Score

Compares the predicted action sequence with the ground-truth action sequence, focusing on the ordering of detected actions.

### Boundary Error

Measures the temporal difference between predicted and ground-truth segment boundaries.

These metrics provide complementary information about both **action classification** and **temporal segmentation**.

---

# Testing

The repository contains automated tests under `tests/`.

### Segmentation Tests

```text
tests/test_segment.py
```

Validates temporal segmentation behaviour.

### Metric Tests

```text
tests/test_metrics.py
```

Validates evaluation metric calculations.

### End-to-End Test

```text
tests/test_pipeline_e2e.py
```

Tests the complete pipeline flow.

Run the test suite with:

```bash
pytest
```

---

# Sample Videos

Sample videos used for local testing or demonstration are stored separately under:

```text
data/sample_videos/
```

Keeping videos inside a dedicated data directory prevents large media files from cluttering the project root.

For larger videos, consider using **Git LFS**, GitHub Releases, or an external download location rather than storing large binary files directly in normal Git history.

---

# Limitations

## 1. Dependence on Text Labels

Zero-shot video-text matching depends on the semantic quality of the supplied labels.

For example:

```text
running
```

is a more direct action description than:

```text
a person moving quickly
```

Carefully chosen labels can therefore affect prediction quality.

## 2. Computational Cost

X-CLIP inference can be computationally expensive, particularly for long or high-resolution videos.

The total processing cost depends on factors such as video length, frame sampling rate, temporal window configuration, hardware, and batch size.

## 3. Limited Temporal Context

The model analyzes temporal clips rather than maintaining unrestricted context over the entire video.

Actions that require longer-range context may therefore be harder to distinguish when individual clips look visually similar.

---

# Key Features

- Zero-shot video action segmentation
- Dynamic, user-defined action labels
- X-CLIP-based video-text matching
- Temporal window processing
- Temporal action segmentation
- Streamlit interface
- Command-line execution
- CSV and JSON result export
- Timeline visualization
- Ground-truth evaluation support
- Unit and end-to-end tests

---

# Technologies

- **Python**
- **PyTorch**
- **X-CLIP**
- **OpenCV**
- **Streamlit**
- **NumPy**
- **Matplotlib**
- **Pandas**

---

# Future Improvements

- Improved temporal boundary detection
- Longer-range temporal modelling
- Better prompt engineering or automatic prompt generation
- GPU inference optimization
- Batch processing for multiple videos
- Expanded evaluation datasets
- Additional visualization options
