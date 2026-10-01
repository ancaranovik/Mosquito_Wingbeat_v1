# Chạy exp_006_sqrt_weights trên Colab

## Mục tiêu và thay đổi duy nhất

Kiểm tra giảm mức ưu tiên lớp ít trong weighted cross-entropy có giảm dự đoán dư
Squamosus và cải thiện window validation Macro-F1 hay không. Đây là giả thuyết,
không phải nguyên nhân đã xác nhận; không bảo đảm F1 tăng hoặc đạt 70–80%.

Đối chứng chính **exp_004_train_norm**, không phải exp005. Giữ Adam LR **0.001**,
TRAIN-only `train_global_zscore`, 8 cấu hình và mọi thiết lập khác của exp004.
Chỉ thay weights từ `N_train/(4*n_train_class)` thành căn bậc hai của biểu thức đó.

| Thiết lập | Exp004 | Exp006 |
|---|---|---|
| Class-weight formula | N_train/(4*n_train_class) | **sqrt(N_train/(4*n_train_class))** |
| LR / normalization | 0.001 / train_global_zscore | Giữ nguyên |
| Seed / batch / max epochs / patience | 42 / 64 / 50 / 8 | Giữ nguyên |
| Checkpoint selection | Validation Macro-F1 cao nhất | Giữ nguyên |
| Early stopping | Weighted validation loss thấp nhất | Giữ quy tắc, dùng weights mới |
| Augmentation / oversampling / pretraining | Không | Giữ nguyên |
| TEST evaluation | Tắt | Tắt |
| Model / split / caches / frontend | Frozen | Giữ nguyên |

Weights dùng đúng counts của **TRAIN windows**, không tính từ VAL/TEST:

| Nhãn | TRAIN windows | Weights exp004 | Weights exp006 |
|---|---:|---:|---:|
| Arabiensis | 10.457 | 0.540619 | 0.735268 |
| Funestus | 5.128 | 1.102428 | 1.049966 |
| Squamosus | 1.426 | 3.964411 | 1.991083 |
| Culex | 5.602 | 1.009149 | 1.004564 |

Relative Squ/Ara weighting giảm khoảng 7.33 xuống 2.71. Không bỏ weights hoàn toàn.
Giảm false positives có thể làm giảm recall; phải xem cả precision/recall/F1.
Weights mới dùng cho TRAIN loss và weighted VAL loss, nên lịch dừng có thể đổi
như hệ quả của policy. **Không so trực tiếp trị số loss với exp004** vì weights
khác; dùng Macro-F1 và per-class/subgroup để đánh giá.

## Code và kết quả cũ

Protocol mới: `stage04_experiment_v3`, engine `tools/experiment_training_v3.py`.
Baseline/v2 engine, model definitions, frozen split/frontend/cache không sửa.
Runner/reviewer nhận thêm v3; vẫn đọc baseline và experiment v2.
V3 giữ accepted inverse weights trong cache evidence và lưu effective weights
riêng trong protocol/result và `class_weights.json` mỗi run. Normalization fit
TRAIN, biến đổi trong RAM; không ghi lại cache. Không tạo lại data/features.

Notebook chứa output exp005 đã sao lưu nguyên byte tại:

```text
_local_only/notebook_snapshots/before_exp006_20261001_213513/
```

Các kết quả cũ trên Drive giữ nguyên. Bản chuẩn bị exp006 đã xóa output để tránh
nhầm experiment. Xem lại exp004/exp005 bằng tên trong 04C, không bật training.

## 1. Bạn commit/push từ máy local

Tại terminal trong project, chạy đúng bộ file này; không stage CSV tải về,
cache, checkpoints hoặc `_local_only`:

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
git diff --check
$exp006Files = @(
  'README.md',
  'configs/experiments/sqrt_weights_v3.json',
  'tools/experiment_training_v3.py',
  'tools/experiment_runner.py',
  'tools/experiment_review.py',
  'tools/test_experiment_v3.py',
  'notebooks/current/04C_compare_models.ipynb',
  'notebooks/current/04D_experiments.ipynb',
  'docs/EXP_006_SQRT_WEIGHTS.md',
  'docs/EXPERIMENT_V2_GUIDE.md',
  'docs/HUONG_DAN_CHAY_THU_NGHIEM.md',
  'docs/PORTABLE_WORKFLOW.md'
)
git add -- $exp006Files
git diff --cached --stat
git commit -m "Prepare exp_006 sqrt class-weight ablation against exp_004"
git push
```

## 2. Bạn mở 04D trên Colab và chạy setup

1. Chọn GPU runtime. Chạy cell mount Drive + clone/pull GitHub.
2. Nếu kernel đã import tools từ code cũ, restart kernel rồi chạy setup lại.
3. Chạy configure dependencies. Repo phải là `/content/edge-ai`, data root phải
   là `/content/drive/MyDrive/EdgeAI`.
4. `verify_portability.py` là audit toàn bộ, không hỗ trợ training. Nếu đã PASS
   và fixed inputs giữ nguyên, có thể bỏ qua lần quét toàn bộ cho trial này.
   Chạy lại khi fixed inputs thay đổi hoặc cần audit đầy đủ.
5. Chạy GPU smoke và đợi PASS. Không bỏ smoke; runner cũng preflight trước training.
   Smoke không cập nhật trọng số và không kiểm chứng F1 hoặc hiệu quả weights.

Không cần chạy lại 00–03B hay train riêng 04A/04B. Cache bị thiếu hoặc sai phải được
kiểm tra nguồn lưu; không chạy frontend để sửa bằng cách regenerate cache.

## 3. Bạn kiểm tra preview trước khi train

04D đã chọn sẵn:

```python
EXPERIMENT_NAME = "exp_006_sqrt_weights"
EXPERIMENT_CONFIG = "configs/experiments/sqrt_weights_v3.json"
RUN_TRAINING = False
```

Chạy cell khi False: chỉ in kế hoạch/params, chưa tạo experiment và chưa train.
Kiểm tra:

- Reference **exp_004_train_norm**; đủ 8 run, seed 42.
- Protocol **stage04_experiment_v3**.
- LR **0.001**, normalization **train_global_zscore**.
- Policy **sqrt_inverse_frequency**, weights gần như bảng trên.
- `evaluate_test: False` và đúng đường lưu exp006.

Tên/mô tả ở cell điều khiển; params trong JSON config. Không chỉnh learning rate,
weights hoặc seed ở các cell model/03A/03B. Nếu exp004 thực tế dùng tên retry,
sửa `reference_experiment_id` về đúng tên trước commit/push.

## 4. Bạn bật training một lần

Sau smoke PASS và preview đúng, đổi riêng `RUN_TRAINING = True` trong bản điều khiển,
rồi chạy lại cell đó. Không cần push giá trị True. Runner chạy 4 model 03A rồi 4 model 03B
lần lượt; chờ **COMPLETE 8/8**. Nếu có lỗi, dừng và giữ logs/artifacts để kiểm tra.

Runner yêu cầu checkout sạch. Nếu notebook mở trong `/content/edge-ai`, dùng
bản sao ngoài repo trước khi chỉnh/lưu output. VS Code local + kernel Colab chỉnh
notebook local sẽ không làm bẩn checkout Colab. Sau chạy đặt lại False.

Kết quả mới:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_006_sqrt_weights/
```

Tên đã tồn tại bị từ chối, kể cả FAILED/INTERRUPTED. Retry dùng tên mới
`exp_006_sqrt_weights_retry01`; đây là run mới, không phải resume hoặc overwrite.

## 5. Bạn xem kết quả trong 04C

Chỉ đọc exp006 sau COMPLETE 8/8; nếu chưa train thì folder/metadata chưa tồn tại.
04C đã chọn `EXPERIMENT_NAME = "exp_006_sqrt_weights"`.
Giữ `REFERENCE_EXPERIMENT_NAME = ""` để dùng reference exp004 từ metadata.
Nếu retry, chọn cùng tên retry trong 04C. Không bật training để xem kết quả.

04C hiển thị weights/provenance, bảng 8 cấu hình, histories, per-class/confusion,
source scores và TRAIN/VAL subgroup diagnostics. Đánh giá:

- Window validation Macro-F1 từng model so với chính model đó ở exp004.
- Squamosus precision/recall/F1 và false positives; tránh chỉ tăng precision
  bằng cách bỏ sót thêm phần lớn Squamosus.
- Arabiensis/HBN và Culex/LT: improvement tổng có đánh đổi nhóm nào không?
- TRAIN–VAL gap và checkpoint/epochs; không so loss tuyệt đối giữa policy.
- Source F1 là chỉ số phụ; TEST vẫn False, chưa có kết luận MCU/INT8.

Lưu output 04C/04D sau chạy để phân tích trước trial tiếp theo. Giữ nguyên kết quả
cũ và tên riêng cho mỗi lần; params trong `results/protocol.json`, provenance ở
`config/metadata.json`, weights mỗi run ở `results/runs/<run_id>/class_weights.json`.
