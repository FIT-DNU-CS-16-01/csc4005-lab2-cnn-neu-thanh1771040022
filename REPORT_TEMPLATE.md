# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên: Nguyễn Trung Thành
- Lớp: FIT-DNU-CS-16-01
- Repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab2-cnn-neu-thanh1771040022
- W&B project: https://wandb.ai/trungthanhk17/csc4005-lab2-neu-cnn

## 2. Bài toán
Mục tiêu là phân loại ảnh lỗi bề mặt thép NEU-CLS vào 6 lớp: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches.  
Trong Lab 2, so sánh 3 hướng: MLP baseline (Lab 1), CNN from scratch và transfer learning (ResNet18) trên cùng pipeline train/val/test.

## 3. Mô hình và cấu hình
### 3.1. MLP baseline từ Lab 1
- Run dùng để so sánh: `run3_sgd` (Lab 1)
- Cấu hình chính: `optimizer=sgd`, `lr=0.01`, `weight_decay=0.001`, `dropout=0.3`, `epochs=30`, `batch_size=32`, `img_size=64`
- Mô hình: MLP (4096 -> 512 -> 256 -> 64 -> 6)

### 3.2. CNN from scratch
- Run: `cnn_small_baseline`
- Cấu hình chính: `model_name=cnn_small`, `train_mode=scratch`, `optimizer=adamw`, `lr=0.001`, `weight_decay=0.0001`, `dropout=0.3`, `epochs=20`, `batch_size=32`, `img_size=64`, `patience=5`, `augment=True`
- Tham số trainable: 32,614

### 3.3. Transfer learning
- Run: `resnet18_transfer`
- Cấu hình chính: `model_name=resnet18`, `train_mode=transfer`, `optimizer=adamw`, `lr=0.001`, `weight_decay=0.0001`, `dropout=0.3`, `epochs=10`, `batch_size=32`, `img_size=128`, `patience=3`, `augment=True`
- Dùng ResNet18 pretrained, chuẩn hóa ImageNet, train classifier head
- Tham số trainable: 3,078 (tổng tham số: 11,179,590)

## 4. Bảng kết quả
| Model | Train mode | Best Val Acc | Test Acc | Epoch time | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | 51.48% | 47.41% | ~3.50 s/epoch (ước tính từ W&B runtime) | 2,245,830 | Baseline Lab 1, còn yếu với đặc trưng không gian |
| CNN-small | scratch | 94.44% | 94.81% | 3.54 s/epoch | 32,614 | Cải thiện rất mạnh so với MLP |
| ResNet18 | transfer | 96.67% | 96.30% | 19.67 s/epoch | 3,078 | Accuracy cao nhất, hội tụ ổn định |

## 5. Phân tích learning curves
- **CNN-small (scratch):** học nhanh từ epoch 1->3 (val_acc 24.44% -> 89.63%), sau đó dao động theo từng epoch; có spike val_loss ở epoch 13 nhưng phục hồi tốt và đạt vùng tốt nhất ở epoch 14-18.
- **ResNet18 (transfer):** val_acc tăng đều từ 79.63% (epoch 1) lên 96.67% (epoch 10), val_loss giảm ổn định, đường học mượt hơn scratch.
- **So sánh tốc độ:** scratch nhanh hơn theo từng epoch (3.54s vs 19.67s), nhưng transfer đạt accuracy cao hơn với ít epoch hơn.

## 6. Confusion matrix và lỗi dự đoán sai
Ảnh confusion matrix đã lưu tại:
- `outputs/cnn_small_baseline/confusion_matrix.png`
- `outputs/resnet18_transfer/confusion_matrix.png`

Các lỗi dự đoán sai nổi bật (theo confusion matrix):
1. **Inclusion -> Pitted_Surface**: nhầm nhiều nhất ở cả 2 mô hình (CNN: 5 mẫu, Transfer: 5 mẫu).
2. **Scratches -> Inclusion**: vẫn còn xuất hiện (CNN: 4 mẫu, Transfer: 1 mẫu).
3. **Pitted_Surface -> Inclusion / Rolled-in_Scale**: có ở CNN scratch (2 và 2 mẫu), giảm rõ ở transfer.
4. **Crazing -> Patches**: chỉ còn thấy ở transfer (2 mẫu), mức thấp.

=> Cặp lớp khó nhất là **Inclusion vs Pitted_Surface**, do đặc trưng bề mặt tương đối giống nhau ở một số ảnh.

## 7. Kết luận
- **CNN có cải thiện so với MLP không?** Có, rất rõ. CNN-small tăng từ 47.41% lên 94.81% test acc (+47.40 điểm phần trăm).
- **Transfer learning có tốt hơn không?** Có trong thí nghiệm này: ResNet18 transfer cao hơn CNN scratch cả val acc (96.67% vs 94.44%) và test acc (96.30% vs 94.81%).
- **Khi nào nên chọn transfer learning thay vì train from scratch?** Nên chọn transfer learning khi cần tối đa độ chính xác trên dữ liệu không quá lớn và có pretrained backbone phù hợp. Scratch phù hợp khi cần mô hình gọn, train nhanh trên CPU và dễ tùy biến kiến trúc.

W&B runs:
- `cnn_small_baseline`: https://wandb.ai/trungthanhk17/csc4005-lab2-neu-cnn/runs/8q6do2f3
- `resnet18_transfer`: https://wandb.ai/trungthanhk17/csc4005-lab2-neu-cnn/runs/rvsv00gj
