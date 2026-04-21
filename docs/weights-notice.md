# Lưu ý về Model Weights

## Bắt buộc có trước khi chạy demo

Hệ thống cần **2 file weights** trong thư mục `weights/`:

| File | Model | Tác dụng | Không có thì sao |
|------|-------|----------|------------------|
| `weights/best.pt` | YOLOv8 | Phát hiện vùng hư hại (scratch/dent/crack) | Không detect được gì |
| `weights/severity.pth` | ResNet18 | Ước lượng mức độ hư hại (0–100%) | Severity hiện -1.0 (N/A) |

## Thứ tự bắt buộc

```
1. Train YOLO (50 epochs)          → weights/best.pt
2. Dùng YOLO crop ảnh hư hại       → data/severity/images/
3. Gán severity thủ công            → data/severity/labels.csv
4. Train severity model (20 epochs) → weights/severity.pth
5. Chạy demo                        → python main.py --image test.jpg
```

Không thể bỏ qua bước nào. AI cần "học" trước rồi mới "nhận diện" được.

## Tại sao weights không có sẵn trong repo?

- File weights nặng (best.pt ~6-50MB, severity.pth ~44MB)
- Đã gitignored để giữ repo nhẹ
- Mỗi lần train ra kết quả khác nhau tuỳ dataset

## Cách share weights cho team

| Cách | Ưu điểm | Nhược điểm |
|------|---------|-----------|
| Google Drive | Dễ, miễn phí | Phải download thủ công |
| Git LFS | Tích hợp git, tự download khi clone | Cần cài thêm, giới hạn dung lượng miễn phí |
| USB / mạng LAN | Nhanh nhất | Cần gặp trực tiếp |

## Hướng dẫn chi tiết

Xem [Training Guide](training-guide.md) để biết cách train từng model.
