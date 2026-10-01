# Chạy exp_004_train_norm trên Colab

Mục tiêu: kiểm tra normalization có giúp đường validation ổn định và cải thiện Macro-F1 không. Một experiment gồm 8 cấu hình: 03A/03B × Small CNN, DS-CNN, TC-ResNet8, CNN-LSTM, chạy lần lượt.

Đối chứng chính là **exp_002_control_v2**, không phải exp_003. Giữ **Adam LR = 0.001** để chỉ thử một thay đổi: `input_normalization = train_global_zscore`. Kết hợp normalization với LR 0.0003 sẽ là một experiment khác nếu kết quả sau này ủng hộ.

| Thiết lập | exp_002_control_v2 | exp_004_train_norm |
|---|---|---|
| Input normalization | none | **train_global_zscore** |
| Adam LR | 0.001 | **Giữ 0.001** |
| Seed / batch size / max epochs / patience | 42 / 64 / 50 / 8 | Giữ nguyên |
| Checkpoint selection | Validation Macro-F1 cao nhất | Giữ nguyên |
| Early stopping | Validation weighted loss thấp nhất | Giữ nguyên |
| Augmentation / TEST evaluation | Không / tắt | Giữ nguyên |
| Split, cache, kiến trúc, class weights | Frozen / v2 | Giữ nguyên |

## Normalization thực hiện như thế nào

Mỗi frontend có một scalar mean/std tính trên toàn bộ giá trị feature của TRAIN. Không tính riêng từng mẫu hoặc từng frequency bin; không dùng validation/TEST để fit thống kê. Validation dùng cùng mean/std của TRAIN.

Áp dụng `(x - mean) / std` trên buffer riêng trong RAM. Không tái tạo feature, không ghi vào frozen cache. Thống kê, fit split, số mẫu và kiểu tính toán được lưu trong `results/runs/<run_id>/normalization.json`, kèm provenance của run.

## 1. Commit và push từ máy local

04D đã chọn tên/config/mô tả exp_004 và để `RUN_TRAINING = False`. 04C đã chọn cùng tên. Config `configs/experiments/train_norm_v2.json` đã có trong repo.

Trong terminal tại thư mục project:

```powershell
git diff --check
git add -- notebooks/current/04D_experiments.ipynb notebooks/current/04C_compare_models.ipynb docs/EXP_004_TRAIN_NORM.md docs/EXPERIMENT_V2_GUIDE.md docs/HUONG_DAN_CHAY_THU_NGHIEM.md docs/PORTABLE_WORKFLOW.md
git commit -m "Prepare exp_004 TRAIN-only normalization trial against exp_002"
git push
```

Notebook chứa output exp_003 đã được sao lưu nguyên vào `_local_only/notebook_snapshots/before_exp004_20261001_133623/`, ngoài Git. Output cũ được xoá khỏi notebook exp_004 để tránh nhầm lần chạy. Các artifact exp_002 và exp_003 trên Drive vẫn là nguồn kết quả chính.

## 2. Setup và GPU smoke trong 04D

1. Mở 04D mới và kết nối Colab GPU. Nếu runtime đã import tools trước khi pull code mới, restart kernel rồi chạy setup lại.
2. Chạy mount Drive + clone/pull repo, sau đó configure dependencies. Repo phải là `/content/edge-ai`; data phải là `/content/drive/MyDrive/EdgeAI`.
3. Cell `tools/verify_portability.py` chỉ audit toàn bộ, không hỗ trợ training. Nếu fixed data/cache/baseline giữ nguyên và audit đã PASS, có thể bỏ qua lần quét toàn bộ cho trial này. Chạy lại khi fixed inputs thay đổi hoặc cần audit đầy đủ.
4. Chạy GPU smoke và đợi PASS. Smoke xác thực CUDA/cache và forward của 8 cấu hình, không cập nhật trọng số; runner cũng preflight trước training. Smoke không phải phép đánh giá hiệu quả normalization; hiệu quả đó cần xem sau run thật.
5. Chạy cell điều khiển với False để xem trước: đủ 8 run, LR **0.001**, `input_normalization: train_global_zscore`, reference **exp_002_control_v2**, `evaluate_test: False`. Chưa tạo experiment hoặc train ở bước xem trước.

## 3. Chạy thật

Cell điều khiển đã chuẩn bị:

```python
EXPERIMENT_NAME = "exp_004_train_norm"
EXPERIMENT_CONFIG = "configs/experiments/train_norm_v2.json"
RUN_TRAINING = False
```

Sau smoke PASS, đổi riêng `RUN_TRAINING = True` trong bản điều khiển rồi chạy lại cell đó. Không cần push giá trị True, không cần chạy lại 03A/03B hoặc train 04A/04B.

Nếu dùng VS Code local + kernel Colab, chỉnh notebook local để checkout Colab vẫn sạch. Nếu notebook nằm trong `/content/edge-ai`, dùng bản sao ngoài checkout trước khi chỉnh. Runner yêu cầu Git sạch để lưu đúng commit/provenance.

Kết quả mới lưu riêng tại:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_004_train_norm/
```

Đợi COMPLETE 8/8. Runner lưu config, Git commit, runtime, thống kê normalization, checkpoints, history, predictions, diagnostics và logs. Tên đã tồn tại bị từ chối, kể cả FAILED/INTERRUPTED. Nếu cần chạy lại, dùng `exp_004_train_norm_retry01` và chọn cùng tên trong 04C; đây là run mới, không phải resume.

## 4. Xem kết quả trong 04C

Chạy phần đọc kết quả sau khi đủ COMPLETE 8/8. 04C đã chọn exp_004. Giữ `REFERENCE_EXPERIMENT_NAME = ""` để tự so sánh với exp_002 theo metadata đã lưu. Nếu chỉ có smoke/preview, experiment chưa tồn tại nên chưa chạy phần đọc kết quả.

Kiểm tra TEST False, Same runtime và params thực thi, rồi quan sát:

- Validation loss/F1 của 03A có bớt dao động, đặc biệt TC-ResNet8, không chỉ xem epoch tốt nhất.
- Chênh lệch Macro-F1 từng cấu hình với exp_002 và TRAIN eval so với validation.
- Precision/recall/F1 Squamosus; lỗi Arabiensis/Culex và bảng HBN/LT.
- Source F1 là chỉ số phụ; giữ window Macro-F1 làm tiêu chí chính như đã định.

Muốn xem exp_003 hoặc exp_002, chỉ đổi tên trong 04C; không bật training. Có thể nhập exp_003 vào ô reference để xem thêm, nhưng khác cả LR và normalization nên không dùng so sánh đó để quy riêng hiệu quả normalization. Lưu output 04C/04D sau run để đánh giá trước experiment tiếp theo; một seed chưa đủ kết luận cải thiện ổn định.
