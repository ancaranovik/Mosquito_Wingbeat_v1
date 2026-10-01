# Chạy exp_005_norm_lr3e4 trên Colab

## Mục tiêu và đối chứng

Kiểm tra giảm learning rate có cải thiện validation Macro-F1 và độ ổn định khi
đầu vào đã được chuẩn hóa hay không. Một experiment gồm 8 cấu hình:
03A/03B × Small CNN, DS-CNN, TC-ResNet8, CNN-LSTM, chạy lần lượt.

Đối chứng chính là **exp_004_train_norm**. So với exp_004, chỉ đổi LR.
Exp_003 có LR 0.0003 nhưng chưa có normalization, nên không dùng exp_003 để
quy riêng tác động giảm LR trên nền normalization. Một seed chưa chứng minh
cải thiện ổn định và trial này không bảo đảm đạt F1 70–80%.

| Thiết lập | exp_004_train_norm | exp_005_norm_lr3e4 |
|---|---|---|
| Adam LR | 0.001 | **0.0003** |
| Input normalization | train_global_zscore | Giữ nguyên |
| Seed / batch / max epochs / patience | 42 / 64 / 50 / 8 | Giữ nguyên |
| Checkpoint selection | Validation Macro-F1 cao nhất | Giữ nguyên |
| Early stopping | Validation weighted loss thấp nhất | Giữ nguyên |
| Augmentation / TEST evaluation | Không / tắt | Giữ nguyên |
| Class weights, model, split, cache | Frozen / v2 | Giữ nguyên |

Mỗi frontend fit một scalar mean/std từ toàn bộ giá trị features của TRAIN.
Validation dùng thống kê TRAIN; không fit trên validation/TEST. Biến đổi trên
buffer riêng trong RAM; không ghi vào frozen cache. Không cần chạy lại 00–03B,
tạo lại features hoặc train riêng 04A/04B.

## 1. Commit và push từ máy local

Config mới: `configs/experiments/norm_lr3e4_v2.json`. 04D đã chọn tên/config/mô tả
exp_005 và để `RUN_TRAINING = False`; 04C đã chọn cùng tên.
Không sửa engine/model hoặc config exp_004.

Notebook có kết quả exp_004 được sao lưu nguyên byte, kiểm tra SHA256 tại:

```text
_local_only/notebook_snapshots/before_exp005_20261001_145711/
```

Bản chuẩn bị exp_005 đã xóa output để tránh nhầm kết quả. Artifact các experiment
cũ trên Drive vẫn giữ nguyên. Không stage `_local_only`, kết quả hoặc checkpoints.

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
git diff --check
$exp005Files = @(
  'configs/experiments/norm_lr3e4_v2.json',
  'notebooks/current/04C_compare_models.ipynb',
  'notebooks/current/04D_experiments.ipynb',
  'docs/EXP_005_NORM_LR3E4.md',
  'docs/EXPERIMENT_V2_GUIDE.md',
  'docs/HUONG_DAN_CHAY_THU_NGHIEM.md',
  'docs/PORTABLE_WORKFLOW.md'
)
git add -- $exp005Files
git diff --cached --stat
git commit -m "Prepare exp_005 LR trial with TRAIN normalization against exp_004"
git push
```

## 2. Setup, smoke và preview trong 04D

1. Chọn Colab GPU, mở 04D mới. Nếu runtime đã import tools trước khi pull code,
   restart kernel rồi chạy setup lại để tránh dùng module cũ.
2. Chạy mount Drive + clone/pull repo, sau đó configure dependencies.
   Repo phải là `/content/edge-ai`, data phải là `/content/drive/MyDrive/EdgeAI`.
3. `tools/verify_portability.py` là audit toàn bộ, không hỗ trợ training. Nếu audit
   đã PASS và fixed data/cache/baseline giữ nguyên, có thể bỏ qua lần quét toàn bộ
   cho trial chỉ đổi LR này. Chạy lại khi fixed inputs đổi hoặc cần audit đầy đủ.
4. Chạy GPU smoke và đợi PASS. Smoke kiểm tra CUDA/cache và forward của 8 cấu hình,
   không cập nhật trọng số; runner vẫn preflight trước training.
5. Chạy cell điều khiển với False:

   ```python
   EXPERIMENT_NAME = "exp_005_norm_lr3e4"
   EXPERIMENT_CONFIG = "configs/experiments/norm_lr3e4_v2.json"
   RUN_TRAINING = False
   ```

   Kiểm tra đủ 8 run, LR **0.0003**, normalization **train_global_zscore**,
   reference **exp_004_train_norm**, `evaluate_test: False` và đường lưu exp_005.
   Preview chưa tạo experiment hoặc train.

## 3. Chạy thật

Sau smoke PASS và preview đúng, đổi riêng `RUN_TRAINING = True` trong bản điều
khiển rồi chạy cell đó. Không cần push giá trị True. Nếu mở notebook trong
checkout `/content/edge-ai`, dùng bản sao ngoài checkout trước khi chỉnh/lưu output,
vì runner yêu cầu Git sạch. VS Code local + kernel Colab có thể chỉnh bản local.

Kết quả mới lưu riêng:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_005_norm_lr3e4/
```

Đợi **COMPLETE 8/8**. Config/metadata, commit, runtime, params, normalization,
checkpoints, history, predictions, diagnostics và logs được lưu theo experiment.
Sau chạy đặt lại False. Tên đã tồn tại bị từ chối, kể cả FAILED/INTERRUPTED.
Retry dùng tên mới, ví dụ `exp_005_norm_lr3e4_retry01`, và chọn cùng tên trong 04C;
đây là run mới, không phải resume.

## 4. Đọc và so sánh trong 04C

Chỉ đọc exp_005 sau COMPLETE 8/8. 04C đã chọn `exp_005_norm_lr3e4`.
Giữ `REFERENCE_EXPERIMENT_NAME = ""` để đọc reference exp_004 từ metadata.
Nếu lần exp_004 thực tế dùng tên retry, cập nhật reference trong config trước
commit/push về đúng tên đó.

Kiểm tra provenance/runtime và TEST False; đánh giá từng model trên:

- **Window validation Macro-F1**: tiêu chí chính; so trực tiếp với exp_004.
- Precision/recall/F1 Squamosus và Culex, không chỉ nhìn điểm tổng.
- Đường loss/F1, epoch được chọn và epoch min-loss; F1 còn tăng lúc dừng không?
- TRAIN eval so với validation và bảng species × HBN/LT.
- Source Macro-F1: chỉ số phụ, không thay thế window Macro-F1.

Mốc exp_004 tốt nhất: window Macro-F1 60.38% với 03A TC-ResNet8;
source Macro-F1 67.45% với 03A DS-CNN. Hai mốc thuộc hai đơn vị đánh giá khác nhau.
Lưu notebook 04C/04D sau run để đánh giá trước trial tiếp theo.

Muốn xem lần cũ, đổi tên trong 04C thành `exp_004_train_norm`, `exp_003_lr3e4`
hoặc `exp_002_control_v2`. Params nằm trong `config/experiment.json` và
`results/protocol.json`; provenance/tiến độ nằm trong `config/metadata.json`.
Không bật training để xem lại kết quả.
