# Hướng dẫn Setup & Flow dự án

## Tổng quan

Hệ thống phát hiện hư hại xe (scratch, dent, crack) từ ảnh và ước lượng mức độ nghiêm trọng (0–100%).

## Flow hoạt động

```
                    ┌──────────────┐
                    │  Ảnh xe đầu  │
                    │    vào       │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   YOLOv8 Detection     │
              │   (weights/best.pt)    │
              │                        │
              │  Quét toàn bộ ảnh →    │
              │  Tìm vùng hư hại →    │
              │  Trả về:               │
              │  • Bounding box (x,y)  │
              │  • Loại: scratch/      │
              │    dent/crack          │
              │  • Confidence %        │
              └────────┬───────────────┘
                       │
            ┌──────────┴──────────┐
            │  Mỗi vùng hư hại   │
            │  được crop ra       │
            └──────────┬──────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │   ResNet18 Severity    │
              │   (weights/severity.   │
              │    pth)                │
              │                        │
              │  Input: crop 224x224   │
              │  Output: severity      │
              │  0–100%                │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │   Visualization        │
              │                        │
              │  Vẽ lên ảnh gốc:       │
              │  • Khung vàng=scratch   │
              │  • Khung cam=dent       │
              │  • Khung đỏ=crack       │
              │  • Label: loại + conf   │
              │    + severity %         │
              └────────┬───────────────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
          ┌───────────┐ ┌──────────┐
          │ Hiển thị  │ │  MQTT    │
          │ / Lưu ảnh │ │ (tuỳ    │
          │ kết quả   │ │  chọn)  │
          └───────────┘ └──────────┘
```

## Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị cho training, không bắt buộc cho inference)
- ~2GB RAM trống

## Cài đặt

### Bước 1: Clone project

```bash
git clone <repo-url>
cd vision-detect
```

### Bước 2: Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows
```

### Bước 3: Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị model weights

Đặt 2 file vào thư mục `weights/`:

| File | Mô tả | Cách có |
|------|--------|---------|
| `best.pt` | YOLO model detect hư hại | Train trên dataset (xem `training-guide.md`) |
| `severity.pth` | CNN model ước lượng severity | Train bằng `train_severity.py` |

## Chạy inference (demo)

### Hiển thị kết quả lên màn hình

```bash
python main.py --image path/to/damaged_car.jpg
```

→ Mở cửa sổ OpenCV hiển thị ảnh + bounding box + label. Nhấn phím bất kỳ để đóng.

### Lưu kết quả ra file

```bash
python main.py --image path/to/damaged_car.jpg --output result.jpg
```

→ Lưu ảnh annotated vào `result.jpg`. Phù hợp khi không có GUI (SSH, server).

### Tuỳ chỉnh confidence threshold

```bash
python main.py --image car.jpg --confidence 0.5
```

→ Chỉ hiện detection có confidence > 50%.

### Chỉ định model weights khác

```bash
python main.py --image car.jpg \
  --yolo-weights path/to/custom_best.pt \
  --severity-weights path/to/custom_severity.pth
```

### Bật MQTT

```bash
python main.py --image car.jpg --mqtt --mqtt-broker 192.168.1.100
```

→ Gửi JSON kết quả qua MQTT topic `vehicle/damage`.

## Kết quả mẫu

### Console output

```
=== Vehicle Damage Detection Results ===
Image: my_car.jpg
Detections: 2

[1] scratch  (conf: 87.2%)  Severity: 34.5%  bbox: [120, 80, 340, 210]
[2] dent     (conf: 92.1%)  Severity: 67.8%  bbox: [400, 150, 550, 300]
========================================
```

### Ảnh output

Ảnh gốc được vẽ thêm:
- **Khung màu** bao quanh vùng hư hại (vàng/cam/đỏ tuỳ loại)
- **Label** phía trên khung: `scratch 87% | Sev: 34.5%`

### MQTT payload (nếu bật)

```json
{
  "damage_type": "scratch",
  "severity": 34.5,
  "confidence": 0.872,
  "timestamp": "2026-04-21T04:00:00+00:00"
}
```

## Cấu trúc thư mục

```
vision-detect/
├── models/
│   ├── yolo_model.py          # Wrapper YOLOv8 — phát hiện vùng hư hại
│   └── severity_model.py      # ResNet18 — ước lượng mức độ (0-100%)
├── utils/
│   ├── preprocessing.py       # Load ảnh, crop vùng bbox
│   └── visualization.py       # Vẽ bbox + label lên ảnh
├── mqtt/
│   └── mqtt_client.py         # Gửi kết quả qua MQTT broker
├── data/severity/             # Dataset training severity
│   ├── images/                # Ảnh crop vùng hư hại
│   └── labels.csv             # Nhãn severity (filename, score)
├── weights/                   # Model weights (best.pt, severity.pth)
├── main.py                    # Pipeline chính
├── train_severity.py          # Script train severity model
└── requirements.txt           # Python dependencies
```

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `YOLO model not found` | Thiếu `weights/best.pt` | Train YOLO hoặc download pretrained model |
| `Severity model not found` | Thiếu `weights/severity.pth` | Train bằng `train_severity.py` |
| `Cannot read image` | Sai đường dẫn ảnh | Kiểm tra lại path, đảm bảo file tồn tại |
| Cửa sổ không hiện | Chạy trên server không GUI | Dùng `--output result.jpg` thay vì hiển thị |
| MQTT connection failed | Broker không chạy | Kiểm tra broker address/port hoặc bỏ flag `--mqtt` |
