# Chạy exp_007_power075 trên Colab

## Mục tiêu

Kiểm tra mức class weights trung gian có giữ phần cải thiện của exp006 đồng thời
phục hồi recall Squ hay không. Đây là giả thuyết, chưa bảo đảm F1 tăng hoặc đạt 70–80%.

Thay đổi duy nhất: weighted CE dùng `(N_train/(4*n_train_class))**0.75`.
Đối chứng chính là exp_004_train_norm (exponent 1); đối chứng bổ sung là
exp_006_sqrt_weights (exponent 0.5). Không đổi sang Focal Loss/Class-Balanced CE.

| Thiết lập | exp007 |
|---|---|
| Experiment ID | exp_007_power075 |
| Config | configs/experiments/power075_v4.json |
| Protocol / engine | stage04_experiment_v4 / tools/experiment_training_v4.py |
| Class-weight policy / exponent | power_inverse_frequency / 0.75 |
| Optimizer / LR | Adam / 0.001 |
| Normalization | train_global_zscore, một mean/std mỗi frontend, fit TRAIN |
| Seed / batch / max epochs / patience | 42 / 64 / 50 / 8 |
| Checkpoint | Validation Macro-F1 cao nhất, hòa chọn epoch sớm nhất |
| Early stopping | Weighted validation CE thấp nhất |
| Models | small_cnn, ds_cnn, tc_resnet8, cnn_lstm |
| Frontends | 03A và 03B, tổng 8 run chạy lần lượt |
| TEST / augmentation / oversampling / pretraining | Tắt / không / không / không |

Weights dùng đúng số TRAIN windows, không tính từ VAL/TEST:

| Lớp | TRAIN windows | exp004 | exp006 | exp007 |
|---|---:|---:|---:|---:|
| Ara | 10.457 | 0.540619 | 0.735268 | 0.630476 |
| Fun | 5.128 | 1.102428 | 1.049966 | 1.075877 |
| Squ | 1.426 | 3.964411 | 1.991083 | 2.809532 |
| Cpx | 5.602 | 1.009149 | 1.004564 | 1.006854 |

Weighted loss vẫn là `sum(weight[label]*NLL)/sum(weight[label])`.
Weights mới dùng cho TRAIN và weighted VAL loss, nên lịch dừng có thể đổi.
Không so trực tiếp giá trị loss giữa các policy; so F1, precision/recall và source.
Không có một ngưỡng F1 tự động quyết định PASS: smoke kiểm tra kỹ thuật,
chất lượng model được đánh giá sau khi train.

## Bảo toàn kết quả cũ

Engine baseline/v2/v3, model definitions, frozen split/frontend/cache và cấu hình
exp004/exp006 giữ nguyên byte. V4 riêng chỉ bổ sung exponent đã khai báo.
Weights gốc và effective weights được lưu cùng exponent trong protocol, result và
class_weights.json mỗi run; normalization thực hiện trong RAM, không ghi cache.

Notebook có output exp006 đã được sao lưu nguyên byte tại:

```text
_local_only/notebook_snapshots/before_exp007_20261002_005648/
```

Notebook hiện tại chọn exp007, training False và xóa output cũ để không nhầm kết quả.
Để xem exp006, mở bản sao hoặc đặt tên exp006 trong 04C. Các experiment trên Drive
không bị sửa. Đọc exp007 sau khi nó COMPLETE 8/8; trước đó metadata chưa tồn tại.

## 1. Bạn commit và push từ local

Chạy tại terminal PowerShell trong project. Chỉ stage các file dưới đây:

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
$exp007Files = @(
  'README.md',
  'configs/experiments/power075_v4.json',
  'tools/experiment_training_v4.py',
  'tools/experiment_runner.py',
  'tools/test_experiment_v4.py',
  'tools/test_experiment_v3.py',
  'notebooks/current/04C_compare_models.ipynb',
  'notebooks/current/04D_experiments.ipynb',
  'docs/EXP_007_POWER075.md',
  'docs/EXPERIMENT_V2_GUIDE.md',
  'docs/HUONG_DAN_CHAY_THU_NGHIEM.md',
  'docs/PORTABLE_WORKFLOW.md'
)
git diff --check
git add -- $exp007Files
git diff --cached --stat
git commit -m "Prepare exp_007 power 0.75 class-weight ablation"
git push
```

Không commit CSV tải về, checkpoint/cache hoặc _local_only. RUN_TRAINING phải False
ở bản commit. Assistant chưa thực hiện commit/push hoặc train.

## 2. Bạn mở 04D, chạy setup và smoke

1. Chọn GPU runtime; chạy mount Drive + clone/pull repo và configure dependencies.
2. Nếu kernel đã import tools của code cũ, restart kernel và chạy setup lại.
3. Kiểm tra repo `/content/edge-ai`, data root `/content/drive/MyDrive/EdgeAI`.
4. Có thể bỏ cell audit full `tools/verify_portability.py` nếu lần trước đã PASS
   và fixed inputs không đổi. Chạy lại khi đổi fixed inputs, gặp hash/path lỗi,
   hoặc cần kiểm tra toàn bộ. Không bỏ GPU smoke.
5. Chạy cell GPU smoke, đợi PASS. Nếu lỗi, dừng để kiểm tra; không train tiếp.

Không cần chạy lại 00–03B hoặc train 04A/04B. Cache thiếu/sai phải kiểm tra nguồn
lưu; không regenerate để vượt gate. Smoke dùng forward kiểm tra shape/CUDA/cache,
không cập nhật trọng số hoặc chứng minh F1 tốt; runner cũng preflight trước train.

## 3. Bạn chạy preview False

Cell điều khiển 04D đã chọn:

```python
EXPERIMENT_NAME = "exp_007_power075"
EXPERIMENT_CONFIG = "configs/experiments/power075_v4.json"
RUN_TRAINING = False
```

Chạy cell, kiểm tra đủ 8 run, reference exp004, policy power_inverse_frequency,
exponent 0.75, LR0.001, TRAIN zscore, TESTFalse và weights đúng bảng trên.
Preview không tạo folder experiment và không train.

Tên/mô tả chỉnh ở cell điều khiển. Exponent và các params nằm trong JSON config.
Không chỉnh params ở 03A/03B hoặc cell model. Nếu cần đổi params, chuẩn bị config
và commit/push trước; lần này giữ nguyên toàn bộ config đã giao.

## 4. Bạn bật True để train một lần

Sau smoke PASS và preview đúng, chỉ đổi RUN_TRAINING thành True rồi chạy cell đó.
Không cần push True. Giữ checkout Colab sạch: nếu notebook nằm trong /content/edge-ai,
dùng bản sao ngoài repo trước khi chỉnh/lưu output. Notebook local trong VS Code
kết nối kernel Colab có thể chỉnh local mà không làm bẩn checkout Colab.

Chờ COMPLETE 8/8. Nếu lỗi, giữ logs/artifacts và gửi traceback; không chạy lại cùng ID.
Tên đã tồn tại, kể cả FAILED/INTERRUPTED, bị từ chối. Retry dùng tên mới như
exp_007_power075_retry01; đó là run mới, không resume hoặc overwrite.
Sau khi train, đặt RUN_TRAINING lại False và lưu output notebook.

Kết quả mới nằm tại:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_007_power075/
```

- config/experiment.json: config của lần chạy.
- config/metadata.json: commit, runtime, trạng thái và danh sách 8 run.
- results/protocol.json: params thực thi, cache identities, weights, implementation hashes.
- results/runs/<run_id>/: history, predictions, metrics, diagnostics, normalization,
  class_weights.json có exponent và effective float32 weights.
- checkpoints/<run_id>/best_model.pt: checkpoint được chọn.
- logs/console.log: log bền vững của experiment.

## 5. Bạn mở 04C để đánh giá

Sau COMPLETE 8/8, chọn EXPERIMENT_NAME cùng tên exp007 (hoặc tên retry).
Giữ REFERENCE_EXPERIMENT_NAME="" để đọc exp004 từ metadata và
ADDITIONAL_REFERENCE_EXPERIMENT_NAME="exp_006_sqrt_weights" để so thêm exp006.
Có thể để tên bổ sung trống khi chỉ cần một đối chứng; cell không so với chính nó.

04C đọc/xác thực artifact rồi hiển thị weights/exponent, histories, confusion,
per-class, source và subgroup. Đọc thêm hai reference cần kiểm tra artifact của
cả hai experiment trên Drive nên có thể mất thời gian.

Đánh giá window Macro-F1 từng cấu hình, Squ precision/recall/F1, số Squ source đúng,
Cpx LT, Ara HBN và khoảng cách TRAIN–VAL. Source F1 khác window F1.
Không chọn chỉ theo điểm tổng khi Squ suy giảm; không so absolute weighted loss.
Không có TEST score hoặc kết luận MCU/INT8 trong lần tuning này.

Lưu output 04C/04D rồi gửi để review. Khi cần đối chiếu sâu, lấy train_eval_predictions.csv
và validation_predictions.csv từ results/runs/<run_id>, giữ riêng theo experiment/model.
