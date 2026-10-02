# Chạy exp_010_time_mask

Exp010 thử đúng một thay đổi: **time masking nhẹ khi train DS-CNN/03A**, so với run DS-CNN/03A của exp007. Không train tám cấu hình. Mục tiêu là kiểm tra độ bền của model một-window khi một đoạn đặc trưng ngắn bị thiếu; đây là giả thuyết cần kiểm chứng, không bảo đảm F1 tăng.

## Cấu hình

| Thành phần | Exp010 |
|---|---|
| Experiment ID | `exp_010_time_mask` |
| Config | `configs/experiments/time_mask_v6.json` |
| Protocol | `stage04_experiment_v6` |
| Kế hoạch | **1 run: `03A_ds_cnn_seed42`** |
| Đối chứng | `exp_007_power075 / 03A_ds_cnn_seed42` |
| Augmentation | Xác suất 0,5; một mask rộng 1–8 frame, vị trí ngẫu nhiên, trên toàn tần số |
| Giá trị mask | 0 **sau** TRAIN-global z-score; tương ứng trung bình TRAIN trong feature gốc |
| Nơi áp dụng | Chỉ loader TRAIN có shuffle dùng để cập nhật model |
| TRAIN eval / validation | Không mask |
| RNG augmentation | Seed42, generator riêng; không dùng RNG model hoặc generator shuffle |
| Adam / LR / weight decay | Như exp007: Adam / 0,001 / **0** |
| Class weight | TRAIN inverse frequency, exponent **0,75** |
| Seed / batch / max epochs / patience | 42 / 64 / 50 / 8 |
| Chọn checkpoint / dừng | F1 validation cao nhất / weighted validation loss |
| TEST | **Tắt và bị chặn trong protocol v6** |
| Split, frontend, model, cache | Giữ nguyên |

Mask được tạo trên tensor đã copy từ cache read-only. Không viết vào `features.npy`, không tạo lại features, không thay WAV hoặc split. Mỗi epoch tiếp tục trạng thái generator augmentation, không reset lại mask về cùng một chuỗi mỗi epoch. Có thể gặp lại một mask do ngẫu nhiên. Thứ tự shuffle giống recipe exp007 khi các điều kiện runtime giống nhau.

Engine v6 mới giữ nguyên các hàm tính loss/metric, model, normalization, optimizer, chọn checkpoint và stopping của v4. Khác biệt training config chỉ có phiên bản protocol và augmentation. Các engine/config của exp007, exp008 giữ nguyên.

## 1. Commit và push trên máy local

Notebook đã backup trước chỉnh sửa tại `_local_only/notebook_snapshots/before_exp010_<timestamp>/`. Output exp009 vẫn giữ trong mục riêng đầu 04C. Các output exp008 của cell được chuyển sang exp010 đã được xóa khỏi notebook mới để không hiểu nhầm là kết quả exp010; bản cũ và artifact Drive vẫn giữ nguyên.

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
git diff --check
git add -- configs/experiments/time_mask_v6.json tools/experiment_training_v6.py tools/experiment_runner.py tools/experiment_review.py tools/test_experiment_v6.py tools/test_experiment_v5.py tools/test_notebook_runtime.py notebooks/current/04C_compare_models.ipynb notebooks/current/04D_experiments.ipynb docs/EXP_010_TIME_MASK.md
git diff --cached --stat
git commit -m "Prepare exp010 single-run TRAIN time-mask ablation against exp007"
git push
```

Không stage `_local_only`, data, cache, checkpoint hoặc artifact experiment. Assistant không train hoặc push trong lần chuẩn bị này. Bản commit phải giữ `RUN_TRAINING=False`.

## 2. Chạy trên Colab

1. Mở 04D từ repo mới, chọn GPU. Nếu dùng kernel cũ, restart trước khi chạy setup để tránh module cũ còn trong bộ nhớ.
2. Chạy cell mount Drive + clone/pull repo, rồi cell configure dependencies.
3. Có thể bỏ **full audit `verify_portability.py`** nếu fixed inputs đã được audit PASS và không thay đổi. Không cần chạy lại 00–03B, 04A hoặc 04B.
4. Chạy **smoke**: lần này cell có `--config configs/experiments/time_mask_v6.json` và kiểm tra đúng DS-CNN/03A. Smoke vẫn xác thực caches theo contract chung, nên vẫn cần cả hai cache đã staging; chỉ có một model forward giả, không training.
5. Chạy cell điều khiển với:

```python
EXPERIMENT_NAME = "exp_010_time_mask"
EXPERIMENT_CONFIG = "configs/experiments/time_mask_v6.json"
RUN_TRAINING = False
```

Preview phải in protocol **v6**, augmentation `train_time_mask`, mask p=0,5 / 1–8 frame, LR=0,001, weight_decay=0, class-weight exponent=0,75, TEST=False và **kế hoạch một run**.

6. Sau smoke PASS và preview đúng, đổi **chỉ `RUN_TRAINING=True`** trong bản notebook Colab rồi chạy cell điều khiển. Đó là lúc training thực sự bắt đầu.

Runner yêu cầu source/config trong checkout đã commit, xác thực selected run exp007 và kiểm tra recipe chỉ khác augmentation, rồi chạy preflight cache/CUDA trước khi tạo thư mục train. Nếu gate lỗi, dừng và gửi traceback; không sửa cache để vượt gate. Notebook UI trên Colab dùng tên/description override nên không cần sửa tracked config trong checkout.

Mỗi tên chỉ dùng một lần. Nếu `exp_010_time_mask` đã tồn tại, xem kết quả cũ hoặc dùng `exp_010_time_mask_retry01` cho lần chạy mới; giữ lần thất bại để kiểm tra. Không ghi đè và không tự resume optimizer.

## 3. Kết quả và cách xem

Colab lưu toàn bộ output tại:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_010_time_mask/
  config/experiment.json
  config/metadata.json
  results/protocol.json
  results/comparison.csv
  results/comparison.json
  results/runs/03A_ds_cnn_seed42/
  checkpoints/03A_ds_cnn_seed42/
  logs/console.log
```

Metadata ghi **1/1** khi hoàn tất, commit/runtime, reference evidence, hash input/checkpoint và config hiệu lực. Run lưu history, checkpoint tốt nhất theo F1/loss, predictions TRAIN eval/validation không mask, confusion matrices, class weights, normalization và diagnostics. Exp007 và exp009 vẫn ở thư mục riêng của chúng. Local cũng dùng central path layer; không ghi kết quả vào Git.

Mở 04C mới, chạy cell bootstrap của phần **training review** rồi cell có:

```python
EXPERIMENT_NAME = "exp_010_time_mask"
```

Sau đó chạy cell history/confusion và đối chiếu. 04C nhận đúng plan một run và vẽ đúng một model; cell đối chiếu dùng run tương ứng trong suite exp007. Các mục exp009 đầu notebook là diagnostic riêng, không cần chạy để xem exp010. Không đổi `DIAGNOSTIC_NAME` của exp009 thành exp010.

Nếu dùng tên retry, nhập cùng tên đó vào 04C. Các experiment suite cũ vẫn đọc được khi nhập tên cũ.

## 4. Cách quyết định sau kết quả

So sánh **Window macro-F1 exp010 với 60,97% của exp007 DS-CNN/03A** trên toàn validation. Không dùng 64,90% của exp009 làm đối chứng vì exp009 là quyết định hai-window. Xem thêm Squ precision/recall, các lớp khác, Source F1, TRAIN eval và khoảng cách TRAIN–VAL.

Gợi ý gate phát triển đã nêu: tăng ít nhất khoảng 2 điểm Window F1 và không giảm Squ recall quá 5 điểm thì xem là candidate cần xác nhận thêm seed. Không chọn seed tốt nhất, không tăng augmentation ngay khi chưa đọc kết quả. Nếu TRAIN/VAL đều giảm, bác bỏ hướng này; nếu gần như không đổi, kiểm tra khả năng fit TRAIN nhỏ trước khi đổi frontend. TEST tiếp tục tắt trong giai đoạn quyết định.

## Kiểm tra local không train

```text
python -B -m unittest discover -s tools -p test_experiment_v6.py -v
python -B tools/experiment_runner.py smoke --device cpu --config configs/experiments/time_mask_v6.json
```

Các test lifecycle dùng artifact tổng hợp và chặn optimizer/inference thật; chúng không phải kết quả scientific. CPU smoke dùng input zero giả, không cập nhật model. Chưa xác nhận GPU training Colab cho exp010 trong lần chuẩn bị.
