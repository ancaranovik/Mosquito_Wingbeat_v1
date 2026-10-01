# Chạy và theo dõi các experiment v2

**Lần tiếp theo: exp_005_norm_lr3e4**, làm theo [hướng dẫn exp_005](EXP_005_NORM_LR3E4.md).
Giữ normalization của exp_004, chỉ giảm LR từ 0.001 xuống 0.0003; đối chứng exp_004_train_norm.
Các bước exp_002 bên dưới được giữ để tra cứu lần đối chứng đã chạy.
04D huấn luyện; 04C đọc kết quả theo tên. Không cần chạy lại 03A/03B hoặc train trong
04A/04B. GitHub lưu code/config, Drive lưu cache cố định và các experiment.

## 1. Những gì thay đổi trong lần 2

| Thiết lập | exp_001 | exp_002_control_v2 |
|---|---|---|
| Model / đầu vào | 4 kiến trúc × 03A, 03B | Giữ nguyên 8 cấu hình |
| Seed, split, class weights | Protocol gốc, seed 42 | Giữ nguyên |
| Optimizer / learning rate | Adam / 0.001 | Giữ nguyên |
| Normalization / augmentation | Không | Giữ nguyên |
| Chọn checkpoint | Validation weighted loss thấp nhất | Validation Macro-F1 cao nhất |
| Early stopping | Validation weighted loss, patience 8 | **Giữ nguyên**, không chuyển sang F1 |
| Checkpoint lưu | Checkpoint được chọn | Cả min-loss và max-F1; `best_model.pt` là max-F1 |
| TRAIN metrics | Online trong lúc học | Online + eval tại checkpoint đã chọn |
| Validation predictions | Không lưu riêng | Lưu theo cửa sổ và theo nguồn |
| TEST | Đánh giá sau selection | Mặc định **tắt khi tuning** |

Giữ early stopping theo loss giúp lần đối chứng chỉ đổi cách chọn checkpoint.
Trong cùng runtime và điều kiện xác định, quá trình cập nhật trọng số của control
tương đương baseline; GPU/package/runtime khác có thể khiến kết quả khác.
Checkpoint F1 cao nhất trong những epoch đã chạy được chọn, ties giữ epoch đầu tiên.
`best_validation_loss` trong result v2 là loss nhỏ nhất quan sát được;
`selected_validation_metrics.loss` là loss tại checkpoint F1 được chọn. Hai số có thể khác nhau.

Protocol gốc ở `tools/experiment_training.py`, model definitions, data, frontend,
scientific locks và baseline artifacts vẫn được kiểm tra bằng các khóa hiện có.
Engine mới là `tools/experiment_training_v2.py`; không sửa/bỏ khóa baseline để tuning.

## 2. Commit / push code từ máy local

04D trong repo để `RUN_TRAINING = False`. Commit các file code/config/notebook/docs
đã chuẩn bị, rồi push lên nhánh GitHub bạn dùng. Không đưa kết quả, cache, checkpoint
hoặc `_local_only` lên Git. Suggested commit:

```text
feat: add validation-driven eight-model experiments
```

Stage đúng bộ file của thay đổi này, kiểm tra rồi commit/push:

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
$experimentFiles = @(
  'README.md', 'docs/PORTABLE_WORKFLOW.md', 'docs/HUONG_DAN_CHAY_THU_NGHIEM.md',
  'docs/EXPERIMENT_V2_GUIDE.md',
  'notebooks/current/04C_compare_models.ipynb', 'notebooks/current/04D_experiments.ipynb',
  'configs/experiments/control_v2.json', 'configs/experiments/lr3e4_v2.json',
  'configs/experiments/train_norm_v2.json',
  'tools/experiment_runner.py', 'tools/experiment_review.py', 'tools/experiment_training_v2.py',
  'tools/test_experiment_v2.py', 'tools/test_experiment_suite.py', 'tools/test_portability.py'
)
git add -- $experimentFiles
git diff --cached --stat
git commit -m "feat: add validation-driven eight-model experiments"
git push
```

Các notebook chứa kết quả lần 1 trước khi chỉnh đã được sao lưu local ở
`_local_only/notebook_snapshots/before_control_v2_20260930/`.
Kết quả chính của lần 1 tiếp tục ở Drive:
`MyDrive/EdgeAI/experiments/exp_001_First_Run_with_Defult_Params/`.

## 3. Chạy exp_002 trong Colab

1. Dùng GPU runtime và mở bản **04D mới**. Nếu kernel đã import tools từ code cũ,
   restart kernel/runtime session trước khi chạy để tránh giữ module cũ trong bộ nhớ.
2. Chạy từ cell setup đầu: mount Drive, clone hoặc `git pull --ff-only`, cài runtime
   pinned, verify scientific/data/cache/baseline, và GPU smoke. Cần PASS trước training.
3. Cell điều khiển đã đặt sẵn:

   ```python
   EXPERIMENT_NAME = "exp_002_control_v2"
   EXPERIMENT_CONFIG = "configs/experiments/control_v2.json"
   RUN_TRAINING = False
   ```

4. Khi còn False, cell chỉ in tên, reference, params thực thi và kế hoạch đủ 8 cấu hình.
   Kiểm tra output đi vào `/content/drive/MyDrive/EdgeAI/experiments/exp_002_control_v2`.
5. Đổi `RUN_TRAINING = True` ở bản điều khiển rồi chạy cell đó. Theo dõi log
   `[1/8]` đến `[8/8]`; chỉ coi xong khi có `COMPLETE: 8/8`.
6. Đặt lại False sau khi chạy. Nếu cell bị chạy lại với cùng tên, runner từ chối
   ghi đè. Lượt FAILED/INTERRUPTED cũng giữ nguyên; xem log và dùng tên mới cho retry,
   ví dụ `exp_002_control_v2_retry01`.

**Checkout chạy code phải sạch.** Trong VS Code local + kernel Colab, điều khiển
notebook local không sửa checkout `/content/edge-ai`. Nếu mở notebook ngay trong
checkout Colab, dùng bản sao bên ngoài checkout, ví dụ `/content/04D_control.ipynb`,
trước khi đổi True hoặc lưu output. Runner ghi chính xác Git commit thực thi.

CLI tương đương, chỉ chạy khi bạn chủ động muốn training:

```bash
python -u -B tools/experiment_runner.py train \
  --config configs/experiments/control_v2.json \
  --experiment-id exp_002_control_v2
```

## 4. Xem exp_002 trong 04C

Chạy setup của 04C ở kernel của nó, rồi đặt:

```python
EXPERIMENT_NAME = "exp_002_control_v2"
```

Chạy các cell đọc kết quả. Bạn sẽ thấy provenance/config, validation Macro-F1
của 8 cấu hình, TRAIN eval và validation source scores, histories, epoch được chọn
và min-loss epoch, confusion matrices/per-class scores, class × acquisition-method
supports/recalls, BatchNorm running stats và bảng chênh lệch validation so với exp_001.
BatchNorm stats là dữ liệu chẩn đoán, không tự chứng minh BatchNorm có lỗi.

Các config v2 mặc định không đánh giá TEST. 04C sẽ thông báo điều đó và hiển thị
validation / TRAIN eval; ô TEST không có điểm là **chưa đánh giá**, không phải lỗi.
Không dùng điểm TEST exp_001 để quyết định hyperparameters của lần tiếp theo.

`REFERENCE_EXPERIMENT_NAME = ""` lấy reference từ config của experiment đã lưu.
Nếu tên lần 1 hoặc control thực tế khác template, nhập tên chính xác trong cell
reference ở 04C. Reference chưa có/FAILED không ảnh hưởng artifact của experiment
hiện tại, nhưng không thể tạo bảng so sánh đã xác thực với nó.

04C chỉ đọc file và xác thực hashes/predictions; không inference, train hoặc ghi
kết quả mới. Chạy lại 04C không tạo experiment khác.

## 5. Sau khi phân tích lần 2

Các template đã chuẩn bị cho từng thử nghiệm riêng, không tự chạy liên tiếp:

| Experiment | Config trong cell 04D | Thay đổi so với exp_002 |
|---|---|---|
| exp_003_lr3e4 | `configs/experiments/lr3e4_v2.json` | Chỉ đổi Adam LR thành 0.0003 |
| exp_004_train_norm | `configs/experiments/train_norm_v2.json` | Chỉ thêm train-global z-score; LR vẫn 0.001 |

Thử nghiệm tiếp theo `exp_005_norm_lr3e4` dùng
`configs/experiments/norm_lr3e4_v2.json`, đối chứng **exp_004_train_norm**:
giữ train-global z-score, chỉ giảm LR từ 0.001 xuống 0.0003. So với exp_002,
exp_005 khác cả LR và normalization; không dùng so sánh đó để quy riêng tác động LR.

Đổi **tên, config và mô tả** trong cùng cell điều khiển 04D. JSON là nơi đặt params:
`training_overrides.learning_rate` hoặc `training_overrides.input_normalization`.
Runner v2 chỉ nhận hai override đã kiểm tra này. Các thay đổi loss, seed, augmentation
hoặc kiến trúc cần một protocol được chuẩn bị riêng; không được âm thầm nhận.

Nếu control thành công dưới tên retry, cập nhật `reference_experiment_id` trong
config tiếp theo về tên đó trước commit/push, hoặc chọn reference đúng trong 04C.
Metadata lưu reference dùng lúc chạy, còn 04C cho phép so sánh bổ sung với tên khác.

Thống kê normalization được tính riêng mỗi frontend từ TRAIN bằng float64, áp dụng
float32 trên buffer tensor riêng khi đọc cache; validation/TEST dùng cùng mean/std.
File cache, features và split không được ghi lại hoặc tạo lại.

TEST chỉ được bật có chủ đích bằng `evaluate_test: true` trong config đã commit,
với tên experiment mới, sau khi chốt phương pháp bằng validation. Không có lệnh
tự đánh giá lại TEST trên experiment cũ và không chọn giữa hai checkpoint theo TEST.

## 6. Tìm lại params và kết quả từng lần

```text
MyDrive/EdgeAI/experiments/<EXPERIMENT_NAME>/
├── config/
│   ├── experiment.json           # tên, mô tả, reference, override và TEST policy
│   └── metadata.json             # commit, runtime, tiến độ và hashes
├── checkpoints/<run_id>/
│   └── best_model.pt             # bản sao checkpoint đã chọn theo F1
├── results/
│   ├── protocol.json             # toàn bộ params thực thi + identities
│   ├── comparison.csv / .json    # chỉ có khi COMPLETE 8/8
│   └── runs/<run_id>/
│       ├── best_model.pt
│       ├── best_val_macro_f1.pt
│       ├── best_val_loss.pt
│       ├── run_spec.json / selection.json / history.json / result.json
│       ├── normalization.json / diagnostics.json / confusion_matrix.json
│       ├── train_eval_predictions.csv / train_eval_source_predictions.csv
│       └── validation_predictions.csv / validation_source_predictions.csv
└── logs/console.log
```

Mỗi lần mới có tên mới. Lần cũ luôn tra lại bằng tên đó trong 04C, kèm params,
Git commit và đường học. Chênh lệch validation trong một seed chưa đủ để tuyên bố
cải thiện chắc chắn; các thử nghiệm nhiều seed sẽ được lên kế hoạch sau.
Trong v2, `confusion_matrix.json` ghi rõ `split`, thứ tự lớp và `matrix`, tránh
nhầm validation với TEST khi experiment tắt TEST.
