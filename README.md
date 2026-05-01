# Vehicle Damage Detection & Severity Estimation

[Tiếng Việt](README.vi.md)

Detect vehicle damage (scratch, dent, crack) using YOLOv8 and estimate severity (0–100%) using a ResNet18 CNN.

## Features

- **YOLO Detection** — scratch, dent, crack with confidence scores
- **Severity Estimation** — ResNet18 regression, 0–100% per damage region
- **Visualization** — bounding boxes + labels + severity overlay
- **MQTT Publishing** — optional JSON output to MQTT broker
- **CLI Interface** — argparse with full configuration
- **Web UI** — Streamlit interactive interface with drag & drop upload
- **PDF Reports** — professional inspection reports with cost estimation
- **Batch Processing** — process image folders with CSV summary export

## Project Structure

```
├── models/
│   ├── yolo_model.py          # YOLOv8 detection wrapper
│   └── severity_model.py      # ResNet18 severity regression
├── utils/
│   ├── preprocessing.py       # Image loading and cropping
│   └── visualization.py       # Bbox drawing and label overlay
├── mqtt/
│   └── mqtt_client.py         # MQTT result publisher
├── data/severity/             # Training dataset directory
│   ├── images/                # Damage crop images
│   └── labels.csv             # filename,severity pairs
├── reports/
│   └── pdf_report.py          # PDF report generator
├── batch/
│   └── batch_processor.py     # Batch image processing
├── weights/                   # Model weights (best.pt, severity.pth)
├── app.py                     # Streamlit web UI
├── main.py                    # CLI inference pipeline
├── train_severity.py          # Severity model training
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Required Weights

Place in `weights/` directory:

- `best.pt` — YOLOv8 model trained on scratch/dent/crack classes
- `severity.pth` — Trained severity regression model

## Usage

### Inference

```bash
# Basic — detect + display
python main.py --image car_photo.jpg

# Save output
python main.py --image car_photo.jpg --output result.jpg

# Custom models + confidence
python main.py --image car_photo.jpg \
  --yolo-weights weights/best.pt \
  --severity-weights weights/severity.pth \
  --confidence 0.3

# With MQTT
python main.py --image car_photo.jpg --mqtt --mqtt-broker 192.168.1.100

# Batch processing
python main.py --input-dir photos/ --output-dir results/

# Batch with PDF reports
python main.py --input-dir photos/ --output-dir results/ --pdf
```

### Web UI

```bash
streamlit run app.py
```

Opens browser at `http://localhost:8501`. Upload image → detect → download annotated image + PDF report.

### Train Severity Model

Prepare dataset in `data/severity/`:

```
data/severity/
├── images/
│   ├── scratch_001.jpg
│   ├── dent_001.jpg
│   └── crack_001.jpg
└── labels.csv
```

`labels.csv` format:

```csv
filename,severity
scratch_001.jpg,25.0
dent_001.jpg,60.0
crack_001.jpg,85.0
```

Run training:

```bash
python train_severity.py \
  --images-dir data/severity/images \
  --labels-csv data/severity/labels.csv \
  --epochs 20 \
  --batch-size 16
```

## Output Example

```
=== Vehicle Damage Detection Results ===
Image: test_car.jpg
Detections: 3

[1] scratch  (conf: 87.2%)  Severity: 34.5%  bbox: [120, 80, 340, 210]
[2] dent     (conf: 92.1%)  Severity: 67.8%  bbox: [400, 150, 550, 300]
[3] crack    (conf: 78.5%)  Severity: 89.2%  bbox: [200, 300, 380, 420]
========================================
```

## MQTT Payload

When `--mqtt` enabled, publishes per detection:

```json
{
  "damage_type": "scratch",
  "severity": 34.5,
  "confidence": 0.872,
  "timestamp": "2026-04-21T03:30:00+00:00"
}
```

## Documentation

- [Setup & Flow Guide](docs/setup-and-flow-guide.md) — Hướng dẫn cài đặt, flow hoạt động, cách chạy demo
- [Training Guide](docs/training-guide.md) — Hướng dẫn train YOLO + severity model từ đầu
- [Weights Notice](docs/weights-notice.md) — Lưu ý về model weights, thứ tự train, cách share cho team

## Tech Stack

- **Detection**: YOLOv8 (Ultralytics)
- **Severity**: ResNet18 (PyTorch/torchvision)
- **Image Processing**: OpenCV
- **MQTT**: paho-mqtt
- **Web UI**: Streamlit
- **PDF**: fpdf2
