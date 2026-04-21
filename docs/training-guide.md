# Hướng dẫn Train AI cho dự án

## Tổng quan

Dự án sử dụng **2 model AI** cần train riêng:

| Model | Framework | Input | Output | File weights |
|-------|-----------|-------|--------|-------------|
| YOLOv8 | Ultralytics | Ảnh xe nguyên | Bbox + class (scratch/dent/crack) | `weights/best.pt` |
| ResNet18 | PyTorch | Ảnh crop vùng hư hại | Severity score 0–100% | `weights/severity.pth` |

---

## Phần 1: Train YOLO Detection Model

### 1.1 Chuẩn bị dataset

**Cách nhanh — dùng Roboflow:**

1. Vào [Roboflow Universe](https://universe.roboflow.com)
2. Search: "vehicle damage detection" hoặc "car damage"
3. Chọn dataset có class: scratch, dent, crack
4. Export format **YOLOv8**
5. Download về, giải nén

**Dataset tự làm:**

Cấu trúc thư mục:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

Mỗi file `.txt` trong `labels/` chứa annotation (1 dòng = 1 object):

```
<class_id> <x_center> <y_center> <width> <height>
```

- Toạ độ normalized (0-1)
- class_id: `0`=scratch, `1`=dent, `2`=crack

**Ví dụ `img_001.txt`:**

```
0 0.45 0.32 0.20 0.15
1 0.70 0.55 0.18 0.22
```

→ 1 scratch + 1 dent trong ảnh.

**File `data.yaml`:**

```yaml
train: ./train/images
val: ./valid/images

nc: 3
names: ['scratch', 'dent', 'crack']
```

**Tool gán nhãn (annotate):**

- [Roboflow](https://roboflow.com) — online, miễn phí
- [LabelImg](https://github.com/HumanSignal/labelImg) — offline
- [CVAT](https://cvat.ai) — online, mạnh

### 1.2 Train YOLO

**Trên máy local (có GPU):**

```bash
yolo detect train \
  data=path/to/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16
```

**Trên Google Colab (miễn phí GPU):**

```python
# Cell 1: Cài ultralytics
!pip install ultralytics

# Cell 2: Upload dataset lên Colab hoặc mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Train
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained
results = model.train(
    data='/content/drive/MyDrive/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
)

# Cell 4: Download best.pt
# Kết quả trong: runs/detect/train/weights/best.pt
```

### 1.3 Các model YOLOv8

| Model | Kích thước | Tốc độ | Độ chính xác |
|-------|-----------|--------|-------------|
| `yolov8n.pt` | 6MB | Nhanh nhất | Thấp nhất |
| `yolov8s.pt` | 22MB | Nhanh | Trung bình |
| `yolov8m.pt` | 50MB | Trung bình | Cao |

**Khuyến nghị cho đề tài:** `yolov8n.pt` hoặc `yolov8s.pt` — nhẹ, train nhanh, đủ demo.

### 1.4 Kiểm tra kết quả YOLO

Sau khi train xong, copy `best.pt`:

```bash
cp runs/detect/train/weights/best.pt weights/best.pt
```

Test nhanh:

```bash
yolo detect predict model=weights/best.pt source=path/to/test_image.jpg
```

### 1.5 Số lượng ảnh khuyến nghị

| Mức | Số ảnh | Kết quả |
|-----|--------|---------|
| Tối thiểu (demo) | 100-200 | Detect được, accuracy thấp |
| Khuyến nghị | 500-1000 | Accuracy tốt |
| Tốt nhất | 2000+ | Production-ready |

---

## Phần 2: Train Severity Model

### 2.1 Chuẩn bị dataset

**Bước 1:** Lấy ảnh crop vùng hư hại

Có 2 cách:

**Cách A — Crop thủ công:** Dùng bất kỳ tool edit ảnh, cắt vùng hư hại ra.

**Cách B — Dùng YOLO crop tự động (sau khi đã có best.pt):**

```python
from models.yolo_model import DamageDetector
from utils.preprocessing import load_image, crop_detection
import cv2

detector = DamageDetector("weights/best.pt")
image = load_image("damaged_car.jpg")
detections = detector.detect(image)

for i, det in enumerate(detections):
    crop = crop_detection(image, det.bbox)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"data/severity/images/{det.class_name}_{i:03d}.jpg", crop_bgr)
```

**Bước 2:** Gán severity score

Mở `data/severity/labels.csv`, gán điểm thủ công:

```csv
filename,severity
scratch_001.jpg,15.0
scratch_002.jpg,30.0
scratch_003.jpg,45.0
dent_001.jpg,50.0
dent_002.jpg,70.0
crack_001.jpg,75.0
crack_002.jpg,90.0
crack_003.jpg,95.0
```

**Quy tắc gán severity (gợi ý):**

| Mức độ | Score | Mô tả |
|--------|-------|-------|
| Nhẹ | 0–30 | Trầy xước nhỏ, không ảnh hưởng kết cấu |
| Trung bình | 30–60 | Móp nhẹ, trầy rõ, cần sửa chữa |
| Nặng | 60–85 | Móp sâu, nứt rõ, hư hại đáng kể |
| Rất nặng | 85–100 | Vỡ nát, biến dạng, cần thay thế |

### 2.2 Train severity model

```bash
python train_severity.py \
  --images-dir data/severity/images \
  --labels-csv data/severity/labels.csv \
  --epochs 20 \
  --batch-size 8 \
  --lr 0.0001
```

**Output mẫu:**

```
2026-04-21 04:00:00 Dataset loaded: 80 samples
2026-04-21 04:00:00 Train: 64 samples | Val: 16 samples
2026-04-21 04:00:00 Backbone frozen for first 10 epochs
2026-04-21 04:00:01 Epoch [1/20] Train Loss: 2500.1234 | Val Loss: 2100.5678
2026-04-21 04:00:02 Epoch [2/20] Train Loss: 1800.4567 | Val Loss: 1500.2345
...
2026-04-21 04:00:10 Backbone unfrozen for fine-tuning
...
2026-04-21 04:00:20 Epoch [20/20] Train Loss: 45.1234 | Val Loss: 52.6789
2026-04-21 04:00:20 Training complete. Best val loss: 48.3456
```

### 2.3 Cách hoạt động training

```
Epoch 1–10 (Transfer Learning):
  ResNet18 backbone ĐÓNG BĂNG (không update weights)
  Chỉ train fc layer (regression head)
  → Model học cách map features → severity score

Epoch 11–20 (Fine-tuning):
  MỞ toàn bộ backbone
  Train tất cả layers với learning rate nhỏ
  → Model tinh chỉnh features cho đúng domain hư hại xe
```

### 2.4 Số lượng ảnh khuyến nghị

| Mức | Số ảnh | Kết quả |
|-----|--------|---------|
| Tối thiểu (demo) | 50–100 | Hoạt động, accuracy thấp |
| Khuyến nghị | 200–500 | Dự đoán tương đối chính xác |
| Tốt nhất | 1000+ | Dự đoán đáng tin cậy |

### 2.5 Tips cải thiện accuracy

- **Cân bằng dataset:** Đảm bảo mỗi mức severity có số ảnh tương đương
- **Data augmentation:** Xoay, flip, thay đổi brightness (thêm vào transform nếu cần)
- **Tăng epochs:** 30–50 nếu dataset lớn
- **Giảm lr:** Thử `0.00005` nếu loss dao động

---

## Phần 3: Quy trình tổng hợp

```
Bước 1: Thu thập ảnh xe hư hại (100+ ảnh)
         │
         ▼
Bước 2: Annotate bằng Roboflow/LabelImg
         (đánh dấu bbox + class)
         │
         ▼
Bước 3: Train YOLO (50 epochs)
         → weights/best.pt
         │
         ▼
Bước 4: Dùng YOLO crop vùng hư hại
         → data/severity/images/
         │
         ▼
Bước 5: Gán severity score thủ công
         → data/severity/labels.csv
         │
         ▼
Bước 6: Train severity model (20 epochs)
         → weights/severity.pth
         │
         ▼
Bước 7: Chạy demo
         python main.py --image test.jpg
```

## Câu hỏi thường gặp

**Q: Train mất bao lâu?**
- YOLO 50 epochs: ~1-2 giờ (Colab GPU), ~6-8 giờ (CPU)
- Severity 20 epochs: ~10-15 phút (CPU), ~2-3 phút (GPU)

**Q: Có cần GPU không?**
- Train: Rất nên có (dùng Colab miễn phí)
- Inference (demo): CPU đủ, mỗi ảnh ~1-2 giây

**Q: Dataset ở đâu?**
- Roboflow Universe: search "vehicle damage", "car damage detection"
- Kaggle: search "car damage dataset"
- Tự chụp: chụp xe hư hại thực tế, đa dạng góc + ánh sáng

**Q: Tại sao dùng ResNet18?**
- Nhẹ (~44MB), train nhanh
- Pretrained ImageNet → tận dụng features sẵn có
- Phù hợp dataset nhỏ (transfer learning)
- Đủ mạnh cho task regression đơn giản
