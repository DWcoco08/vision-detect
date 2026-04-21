# Phát hiện hư hại xe & Ước lượng mức độ nghiêm trọng

[English](README.md)

Hệ thống phát hiện hư hại xe (trầy xước, móp, nứt) từ ảnh bằng YOLOv8 và ước lượng mức độ nghiêm trọng (0–100%) bằng CNN ResNet18.

## Tính năng

- **Phát hiện hư hại** — nhận diện trầy xước (scratch), móp (dent), nứt (crack) kèm độ tin cậy
- **Ước lượng mức độ** — ResNet18 regression, trả về 0–100% cho từng vùng hư hại
- **Trực quan hoá** — vẽ khung bao + nhãn + % severity lên ảnh
- **Gửi MQTT** — tuỳ chọn gửi kết quả JSON qua MQTT broker
- **Giao diện CLI** — hỗ trợ đầy đủ tham số dòng lệnh
- **Giao diện Web** — Streamlit, kéo thả ảnh trực tiếp trên trình duyệt
- **Báo cáo PDF** — xuất báo cáo kiểm tra chuyên nghiệp kèm ước tính chi phí
- **Xử lý hàng loạt** — xử lý thư mục ảnh, xuất CSV tổng hợp

## Cấu trúc dự án

```
├── models/
│   ├── yolo_model.py          # Wrapper YOLOv8 — phát hiện vùng hư hại
│   └── severity_model.py      # ResNet18 — ước lượng mức độ (0-100%)
├── utils/
│   ├── preprocessing.py       # Load ảnh, crop vùng bbox
│   └── visualization.py       # Vẽ bbox + label lên ảnh
├── mqtt/
│   └── mqtt_client.py         # Gửi kết quả qua MQTT broker
├── data/severity/             # Thư mục dataset training
│   ├── images/                # Ảnh crop vùng hư hại
│   └── labels.csv             # Nhãn: filename, severity
├── reports/
│   └── pdf_report.py          # Tạo báo cáo PDF
├── batch/
│   └── batch_processor.py     # Xử lý ảnh hàng loạt
├── weights/                   # File weights model (best.pt, severity.pth)
├── app.py                     # Giao diện web Streamlit
├── main.py                    # Pipeline CLI chính
├── train_severity.py          # Script train severity model
└── requirements.txt           # Dependencies
```

## Cài đặt

### Bước 1: Clone & tạo môi trường

```bash
git clone <repo-url>
cd vision-detect
python -m venv venv
source venv/bin/activate
```

### Bước 2: Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị model weights

Đặt vào thư mục `weights/`:

| File | Mô tả | Cách có |
|------|--------|---------|
| `best.pt` | Model YOLO detect scratch/dent/crack | Train trên dataset hư hại xe |
| `severity.pth` | Model CNN ước lượng severity | Train bằng `train_severity.py` |

> **Quan trọng:** Phải train AI trước khi chạy demo. Xem [Lưu ý về Weights](docs/weights-notice.md).

## Sử dụng

### Chạy demo (inference)

```bash
# Hiển thị kết quả lên màn hình
python main.py --image path/to/xe_hu_hai.jpg

# Lưu ảnh kết quả
python main.py --image path/to/xe_hu_hai.jpg --output ket_qua.jpg

# Tuỳ chỉnh confidence
python main.py --image xe.jpg --confidence 0.5

# Bật MQTT
python main.py --image xe.jpg --mqtt --mqtt-broker 192.168.1.100

# Xử lý hàng loạt
python main.py --input-dir photos/ --output-dir results/

# Hàng loạt + xuất PDF
python main.py --input-dir photos/ --output-dir results/ --pdf
```

### Giao diện Web

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`. Upload ảnh → phát hiện hư hại → tải ảnh annotated + báo cáo PDF.

### Train severity model

Chuẩn bị dataset trong `data/severity/`:

```
data/severity/
├── images/
│   ├── scratch_001.jpg
│   ├── dent_001.jpg
│   └── crack_001.jpg
└── labels.csv
```

Format `labels.csv`:

```csv
filename,severity
scratch_001.jpg,25.0
dent_001.jpg,60.0
crack_001.jpg,85.0
```

Chạy training:

```bash
python train_severity.py \
  --images-dir data/severity/images \
  --labels-csv data/severity/labels.csv \
  --epochs 20 \
  --batch-size 16
```

## Kết quả mẫu

### Console

```
=== Vehicle Damage Detection Results ===
Image: test_car.jpg
Detections: 3

[1] scratch  (conf: 87.2%)  Severity: 34.5%  bbox: [120, 80, 340, 210]
[2] dent     (conf: 92.1%)  Severity: 67.8%  bbox: [400, 150, 550, 300]
[3] crack    (conf: 78.5%)  Severity: 89.2%  bbox: [200, 300, 380, 420]
========================================
```

### Ảnh output

Ảnh gốc được vẽ thêm:
- **Khung vàng** = trầy xước (scratch)
- **Khung cam** = móp (dent)
- **Khung đỏ** = nứt (crack)
- Kèm label: `loại + confidence + severity %`

### MQTT payload

```json
{
  "damage_type": "scratch",
  "severity": 34.5,
  "confidence": 0.872,
  "timestamp": "2026-04-21T04:00:00+00:00"
}
```

## Tài liệu

- [Hướng dẫn Setup & Flow](docs/setup-and-flow-guide.md) — Cài đặt, flow hoạt động, cách chạy demo
- [Hướng dẫn Train AI](docs/training-guide.md) — Train YOLO + severity model từ đầu
- [Lưu ý về Weights](docs/weights-notice.md) — Thứ tự train, cách share weights cho team

## Công nghệ sử dụng

- **Phát hiện vật thể**: YOLOv8 (Ultralytics)
- **Ước lượng mức độ**: ResNet18 (PyTorch/torchvision)
- **Xử lý ảnh**: OpenCV
- **Giao tiếp IoT**: paho-mqtt
- **Giao diện web**: Streamlit
- **Xuất PDF**: fpdf2
