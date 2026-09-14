<img width="1917" height="1020" alt="image" src="https://github.com/user-attachments/assets/c2ebb95f-3fa5-4dbf-8219-f7f3e299cb0e" />



# ⛑️ Helmet Detector

**AI-powered safety helmet detection — with a dashboard you'd actually want to use.**

A dark, card-based Streamlit dashboard for testing a YOLO helmet-detection model on images, videos, webcam snapshots, and live camera feeds.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b)](https://streamlit.io/)
[![YOLOv12n](https://img.shields.io/badge/model-YOLOv12n-00ffcc)](https://github.com/ultralytics/ultralytics)
[![mAP50](https://img.shields.io/badge/mAP50-0.847-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#license)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Detected Classes](#-detected-classes)
- [Model & Training Details](#-model--training-details)
- [Dataset](#-dataset)
- [Evaluation Results](#-evaluation-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Customization](#-customization)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔎 Overview

Helmet Detector wraps a custom-trained [YOLOv12n](https://github.com/ultralytics/ultralytics) model in a polished Streamlit dashboard so you can quickly sanity-check detections without writing any inference code. Drop in `.pt` weights, pick an input source, and see annotated results, confidence scores, and per-detection bounding boxes instantly.

## ✨ Features

| | |
|---|---|
| 🖼️ **Image detection** | Upload a photo — see the original and annotated result side by side |
| 🎬 **Video detection** | Upload a video — processed frame-by-frame with a live preview and progress bar |
| 📸 **Webcam snapshot** | Capture a single photo straight from your camera |
| 📡 **Live webcam** | Real-time detection over your camera feed via WebRTC |
| 📊 **Metrics dashboard** | Total detections, With/Without Helmet counts, and average confidence at a glance |
| 📈 **Detection overview chart** | Visual breakdown of detections per class |
| 📋 **Detection details table** | Per-box class, confidence, and coordinates, exportable to CSV |
| 🕑 **Run history** | Every run logged in-session with timestamp and summary stats |
| ⚙️ **Tunable thresholds** | Adjustable confidence and IoU (NMS) sliders |
| 🎨 **Custom dark UI** | Purpose-built dashboard styling, not default Streamlit chrome |

## 🏷️ Detected Classes

```yaml
names: ['With Helmet', 'Without Helmet']
```

## 🧠 Model & Training Details

The bundled weights (`1789371349153_best__1_.pt`) were trained with [Ultralytics](https://github.com/ultralytics/ultralytics) on a **YOLOv12n** architecture.

| | |
|---|---|
| **Base model** | `yolov12n.pt` (pretrained, fine-tuned) |
| **Ultralytics version** | 8.4.150 |
| **Classes** | 2 — `With Helmet`, `Without Helmet` |
| **Image size** | 640×640 |
| **Epochs (max / actual)** | 100 / **89** (early stopping, patience=20) |
| **Best epoch** | 69 |
| **Optimizer** | AdamW (auto-selected), lr0 ≈ 0.001667, momentum 0.9 |
| **Batch size** | 16 |
| **Augmentations** | Mosaic, HSV jitter, horizontal flip (p=0.5), Blur/MedianBlur/ToGray/CLAHE (Albumentations, p=0.01 each) |
| **Hardware** | Tesla T4 (14.9 GB), CUDA, AMP enabled |
| **Training time** | ~1.7 hours |
| **Params (fused)** | 2,557,118 — 159 layers, 7.3 GFLOPs |
| **Weights size** | ~5.5 MB |
| **Inference speed** | ~7.7 ms/image (GPU) — 0.3 ms preprocess, 54.2 ms first-pass / 7.7 ms steady-state inference, 5.4 ms postprocess |

## 📦 Dataset

Trained on the **[Bike Helmet Detection](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo)** dataset from Roboflow Universe.

| Split | Images | Notes |
|---|---|---|
| Train | 3,546 | includes augmented variants (129 background/negative images) |
| Validation | 126 | 299 labeled instances |
| Test | 63 | held out for final inference sampling |

> If you retrain on your own data, point `data=` in your Ultralytics training command at your own `data.yaml` with the same two class names (or update `CLASS_NAMES` in `app.py` to match).

## 📈 Evaluation Results

Final validation metrics (best.pt, IoU=0.7):

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **All** | 126 | 299 | 0.810 | 0.814 | **0.847** | 0.434 |
| With Helmet | 88 | 184 | 0.864 | 0.880 | 0.917 | 0.476 |
| Without Helmet | 53 | 115 | 0.755 | 0.748 | 0.778 | 0.393 |

**Takeaways:**
- The model is noticeably stronger at detecting **helmeted** riders (mAP50 0.92) than unhelmeted ones (mAP50 0.78) — likely due to class imbalance in training data (184 vs. 115 instances) and the more distinct visual signature of helmets vs. bare heads.
- mAP50-95 (0.434 overall) reflects stricter IoU matching — boxes are reasonably accurate but not pixel-perfect, which is typical for a lightweight `n`-scale model.
- If you need higher recall on "Without Helmet" specifically (the higher-stakes class for a safety application), consider: more negative-class training data, a larger backbone (`yolov12s`/`m`), or lowering the confidence threshold in the app for that use case at the cost of more false positives.

## 📁 Project Structure

```
helmet-detection/
├── app.py                          # Streamlit app (entry point)
├── requirements.txt                # Python dependencies
├── 1789371349153_best__1_.pt       # YOLOv12n model weights (default; optional — can upload at runtime)
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A trained YOLO `.pt` model (Ultralytics format), or use the one included in this repo
- A webcam, if you want to use the snapshot/live-detection modes

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tnithin236/helmet-detection.git
   cd helmet-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints — usually **http://localhost:8501**.

## 🖱️ Usage

1. Open the app in your browser.
2. In the sidebar, confirm the model weights are loaded (the bundled `.pt` file loads automatically, or upload your own).
3. Adjust the **confidence** and **IoU** thresholds if needed.
4. Choose an **input source**:
   - **Image** — upload a `.jpg` / `.png` / `.webp`
   - **Upload Video** — upload a `.mp4` / `.mov` / `.avi` / `.mkv`
   - **Webcam Snapshot** — capture a photo from your camera
   - **Live Webcam** — stream real-time detection from your camera
5. Review the annotated result, metrics, and detection table.
6. Download detections as CSV, or check the **History** page for past runs.

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| Model weights | Sidebar file uploader, or `1789371349153_best__1_.pt` in the project root | bundled model |
| Confidence threshold | Sidebar slider | `0.25` |
| IoU threshold (NMS) | Sidebar slider | `0.45` |
| Class names | `CLASS_NAMES` in `app.py` | `['With Helmet', 'Without Helmet']` |

## ☁️ Deployment

**Streamlit Community Cloud**
1. Push this repo to GitHub (already done if you're reading this from `tnithin236/helmet-detection`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, and select this repo.
3. Set the main file path to `app.py` and deploy.

**Docker** (optional — create a `Dockerfile` like below if you want containerized deployment)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

> **Note:** Live Webcam mode (WebRTC) needs the app served over `https` or accessed via `localhost` — browsers block camera access on plain `http` for remote hosts.

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | Run `pip install -r requirements.txt` |
| Live Webcam mode shows an import error | Run `pip install streamlit-webrtc av` and restart the app |
| Camera doesn't prompt for permission | Make sure you're on `localhost` or `https`, not plain `http` on a remote server |
| Model fails to load | Confirm it's an Ultralytics-format `.pt` checkpoint (YOLOv8/v11/v12-style zip archive) |
| Slow video processing | The app samples frames (~5 fps) by default — adjust `sample_every` in `app.py` |
| Missed "Without Helmet" detections | Expected given the class's lower recall (0.748) — try lowering the confidence threshold slider |

## 🎨 Customization

- **Colors / theme** — edit the `GREEN`, `RED`, `BLUE` constants and the `<style>` block at the top of `app.py`.
- **Class names** — update `CLASS_NAMES` and `SHORT_LABEL` in `app.py` if you retrain with different classes.
- **Default thresholds** — change the `default` values passed to `slider_with_label(...)` in the sidebar section.

## 🗺️ Roadmap

- [ ] Retrain with more "Without Helmet" examples to close the per-class recall gap
- [ ] Persistent history (save across sessions, not just in-memory)
- [ ] Multi-model comparison mode
- [ ] Alerting/webhook integration for "Without Helmet" detections
- [ ] Batch image processing

## 🤝 Contributing

Contributions are welcome! To propose a change:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a pull request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using Streamlit and Ultralytics YOLOv12n — <b>Better Safety Through AI</b></p>
