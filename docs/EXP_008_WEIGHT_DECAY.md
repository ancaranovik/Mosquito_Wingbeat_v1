# Chạy exp_008_power075_wd1e4 trên Colab

Lần này kiểm tra một giả thuyết: Adam L2 regularization có giúp generalize tốt hơn exp007 hay không. Chỉ đổi `weight_decay` từ 0 lên **0.0001 (1e-4)**. Không có bảo đảm F1 tăng. Audit TRAIN theo mẫu không chứng minh nhãn sai hoặc toàn bộ feature mất tín hiệu; không loại file, đổi frontend hoặc tạo lại cache.

## Cấu hình cố định

| Thông số | Exp007 | Exp008 |
|---|---|---|
| Tên | exp_007_power075 | exp_008_power075_wd1e4 |
| Config | power075_v4.json | power075_wd1e4_v5.json |
| Protocol | stage04_experiment_v4 | stage04_experiment_v5 |
| Adam weight_decay | 0 | **0.0001** |
| Adam LR | 0.001 | 0.001 |
| Weighted CE | inverse-frequency từ TRAIN, exponent 0.75 | Giữ nguyên |
| Input normalization | TRAIN-only global scalar z-score | Giữ nguyên |
| Seed / batch / max epochs / patience | 42 / 64 / 50 / 8 | Giữ nguyên |
| Checkpoint selection | Validation window macro-F1, tie lấy epoch sớm nhất | Giữ nguyên |
| Early stopping | Minimum weighted validation loss | Giữ nguyên |
| Frontend / model / split / cache | 03A, 03B × 4 model đã freeze | Giữ nguyên |
| Augmentation | Không | Không |
| TEST | Tắt | Tắt |
| Đối chứng chính | exp004 | **exp007** |

Protocol v5/engine mới giữ nguyên code baseline/v2/v3/v4. Dùng `torch.optim.Adam` như trước, không đổi sang AdamW, không tạo parameter groups: weight_decay áp dụng cho toàn bộ parameters được truyền vào Adam như engine cũ. Weighted CE loss được báo cáo vẫn là CE; không cộng riêng L2 vào loss hiển thị. Early stopping vẫn theo loss validation đó. Kết quả lưu effective config, protocol, hash implementation, commit Git và runtime để phân biệt hai lần.

Một lần train chạy lần lượt **03A: small_cnn, ds_cnn, tc_resnet8, cnn_lstm**, rồi 4 model tương tự của **03B**. Không cần chạy lại 00–03B hoặc train riêng 04A/04B.

Chuẩn bị cục bộ đã PASS 10 tests v5 và 55 tests regression cho suite/v2/v3/v4/storage/notebook runtime. Tests dùng fixtures và mock phần scientific training; không train dataset. 97 code cell notebook parse được; 54 đầu vào audit có hash không đổi; engine v4, config exp007, model/frontend cũ không đổi. Chưa chạy GPU smoke trên Colab cho bản cập nhật này; bạn thực hiện bước 2 trước khi bật True.

## 1. Commit và push từ máy local

Assistant chỉ chuẩn bị files và kiểm tra không training; bạn thực hiện commit/push. Bản commit phải giữ `RUN_TRAINING=False`.

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
$exp008Files = @(
  'configs/experiments/power075_wd1e4_v5.json',
  'tools/experiment_training_v5.py',
  'tools/experiment_runner.py',
  'tools/test_experiment_v5.py',
  'tools/test_experiment_v4.py',
  'notebooks/current/04C_compare_models.ipynb',
  'notebooks/current/04D_experiments.ipynb',
  'docs/EXP_008_WEIGHT_DECAY.md',
  'README.md',
  'docs/HUONG_DAN_CHAY_THU_NGHIEM.md',
  'docs/EXPERIMENT_V2_GUIDE.md',
  'docs/PORTABLE_WORKFLOW.md'
)
git diff --check
git add -- $exp008Files
git diff --cached --stat
git commit -m "Prepare exp008 Adam weight decay ablation against exp007"
git push
```

Không stage Downloads, `_local_only`, CSV, cache, checkpoint hoặc kết quả experiment. Output notebook exp007 trước cập nhật được giữ nguyên tại:

```text
_local_only/notebook_snapshots/before_exp008_20261002_123046/
```

Backup này gồm bản notebook và sha256.json. Artifact exp007 trên Drive vẫn ở `MyDrive/EdgeAI/experiments/exp_007_power075/`; không ghi đè. Muốn xem lại, nhập `EXPERIMENT_NAME="exp_007_power075"` trong 04C hoặc mở snapshot.

## 2. Setup trên Colab

1. Mở 04D mới, chọn GPU; chạy mount Drive, clone/pull GitHub và configure dependencies.
2. Nếu kernel đang dùng module của code cũ, pull rồi restart kernel và chạy setup lại. Preview phải dùng protocol v5, không v4. Không dùng code Windows D:\ trong runtime Colab.
3. Repo phải là `/content/edge-ai`; data root là `/content/drive/MyDrive/EdgeAI`.
4. Có thể bỏ cell `tools/verify_portability.py` khi lần audit trước PASS và fixed inputs không đổi. Đây là full audit, không hỗ trợ training; GPU smoke và preflight cache/hash vẫn cần chạy.
5. Chạy cell GPU smoke, đợi PASS. Nếu lỗi CUDA/hash/cache, dừng và gửi traceback; không regenerate hoặc sửa cache để vượt gate.

Smoke không cập nhật trọng số. Runner cũng chạy preflight trước train. Không cần full audit 15 phút mỗi lần kết nối kernel.

## 3. Preview với False

Cell điều khiển 04D đã đặt:

```python
EXPERIMENT_NAME = "exp_008_power075_wd1e4"
EXPERIMENT_CONFIG = "configs/experiments/power075_wd1e4_v5.json"
RUN_TRAINING = False
```

Chạy cell và kiểm tra:

- Protocol `stage04_experiment_v5`, reference `exp_007_power075`, đủ 8 run.
- `weight_decay=0.0001`, `learning_rate=0.001`, `class_weight_exponent=0.75`.
- `input_normalization=train_global_zscore`, seed 42, batch 64, max_epochs 50, patience 8.
- `evaluate_test=False`; output đúng `.../EdgeAI/experiments/exp_008_power075_wd1e4`.

Preview không tạo thư mục experiment. Tên/mô tả chỉnh trong cell này; params nằm trong JSON trên GitHub. Lần này giữ nguyên config, không đổi thêm LR/weights/epoch.

## 4. Bật True để chạy một lần

Sau smoke PASS và preview đúng, chỉ đổi `RUN_TRAINING=True` trong bản notebook điều khiển, rồi chạy lại cell đó. **Không cần push True**. Checkout `/content/edge-ai` phải sạch vì runner ghi commit đã thực thi: nếu chỉnh/lưu notebook trong checkout, hãy dùng bản sao notebook ngoài checkout. Notebook local kết nối kernel Colab có thể chỉnh local mà không làm bẩn checkout Colab.

Chờ `COMPLETE 8/8`. Không chạy lại cell True khi đã hoàn tất. Nếu có lỗi/timeout, giữ artifacts/log; cùng ID bị từ chối dù FAILED/INTERRUPTED. Retry phải dùng tên mới, ví dụ `exp_008_power075_wd1e4_retry01`; đó là lượt mới, không resume.

Sau train đặt lại False. Xem kết quả bằng 04C mà không bật training. Có thể dùng CLI tương đương (chỉ khi chủ động muốn train):

```sh
python -u -B tools/experiment_runner.py train --config configs/experiments/power075_wd1e4_v5.json --experiment-id exp_008_power075_wd1e4
```

## 5. Nơi lưu và cách so sánh

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_008_power075_wd1e4/
  config/experiment.json        tên, mô tả, params, reference
  config/metadata.json          commit, runtime, trạng thái 8 run, hash
  logs/console.log              log từng epoch và lỗi nếu có
  checkpoints/<run_id>/best_model.pt
  results/protocol.json         effective config, identities, implementation hashes
  results/comparison.csv
  results/comparison.json
  results/runs/<run_id>/        history, checkpoints, metrics, predictions, diagnostics
```

Trong LOCAL mode, khi không đặt `EDGEAI_DATA_ROOT`, output tương ứng ở `_runtime/experiments/<tên>/`. Đặt data root khác thì central path layer dùng `<EDGEAI_DATA_ROOT>/experiments/<tên>/`.

Mở 04C, chạy setup rồi cell chọn `EXPERIMENT_NAME="exp_008_power075_wd1e4"` (hoặc tên retry đã dùng). Sau COMPLETE 8/8, các bảng, histories và confusion matrices đọc theo tên này; không dùng output exp007 cũ. Đối chứng chính tự lấy exp007 từ metadata; đối chứng bổ sung mặc định exp004, có thể để trống.

So với exp007: window macro-F1 là chỉ số chính; xem recall/precision/F1 từng loài, source F1 riêng, TRAIN–VAL gap, subgroup supports/recalls và epoch chọn. Không lấy source F1 thay window F1. Exp004 khác CE weights nên không so trực tiếp loss với exp004. Cùng seed không loại hết bất định; cải thiện nhỏ cần nhiều seed sau khi chọn candidate. TEST không dùng để tune hoặc shortlist.

Nếu reference chưa có trên data root hiện tại, 04C thông báo và vẫn xác thực riêng experiment mới. Giữ nguyên experiment cũ để so sánh; không chép config mới vào folder kết quả cũ.

Khi xong, gửi output 04C/04D. Để phân tích sâu, gửi toàn bộ folder experiment hoặc ít nhất metadata/protocol/comparison, result/history/diagnostics và TRAIN-eval/validation predictions của từng run; không cần gửi cache hoặc raw data.
