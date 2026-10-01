# Chạy exp_003_lr3e4 trên Colab

Mục tiêu: kiểm tra giảm learning rate có giúp đường validation ổn định và Macro-F1 tốt hơn không. Một experiment gồm 8 cấu hình: 03A/03B × Small CNN, DS-CNN, TC-ResNet8, CNN-LSTM, chạy lần lượt.

| Thiết lập | exp_002_control_v2 | exp_003_lr3e4 |
|---|---|---|
| Adam learning rate | 0.001 | **0.0003** |
| Seed / batch size / max epochs / patience | 42 / 64 / 50 / 8 | Giữ nguyên |
| Checkpoint selection | Validation Macro-F1 cao nhất | Giữ nguyên |
| Early stopping | Validation weighted loss thấp nhất | Giữ nguyên |
| Normalization / augmentation | Không / không | Giữ nguyên |
| TEST evaluation | Tắt | Tắt |
| Split, caches, kiến trúc, class weights | Frozen / v2 | Giữ nguyên |

## 1. Commit và push từ máy local

04D đã đặt tên/config/mô tả cho exp_003, với `RUN_TRAINING = False`. 04C đã chọn cùng tên. Config `configs/experiments/lr3e4_v2.json` đã có trong repo; reference là `exp_002_control_v2`.

Trong terminal tại thư mục project:

```powershell
git diff --check
git add -- notebooks/current/04D_experiments.ipynb notebooks/current/04C_compare_models.ipynb tools/test_experiment_suite.py tools/test_experiment_v2.py docs/EXP_003_LR3E4.md docs/EXPERIMENT_V2_GUIDE.md docs/HUONG_DAN_CHAY_THU_NGHIEM.md docs/PORTABLE_WORKFLOW.md
git commit -m "Prepare exp_003 learning-rate trial against exp_002"
git push
```

Output exp_002 đã được lưu nguyên trong `_local_only/notebook_snapshots/before_exp003_20261001_124511/`, ngoài Git. Bản notebook exp_003 được xoá output cũ để tránh nhầm experiment. Các artifact exp_002 trên Drive vẫn là nguồn kết quả chính.

## 2. Setup và smoke test trong 04D

1. Mở 04D mới, kết nối **Colab GPU**. Nếu runtime đã import tools trước khi pull code mới, restart kernel rồi chạy setup lại.
2. Chạy cell mount Drive + clone/pull repo, sau đó cell cấu hình dependencies. Output phải là repo `/content/edge-ai`, data `/content/drive/MyDrive/EdgeAI`.
3. Cell `tools/verify_portability.py` là kiểm tra toàn bộ, không training. Nếu dữ liệu/cache/fixed baseline giữ nguyên và kiểm tra này đã PASS ở exp_002, có thể bỏ qua lần quét toàn bộ cho thử nghiệm chỉ đổi LR này. Chạy lại khi thay dữ liệu/cache/baseline hoặc khi cần audit đầy đủ.
4. Chạy **GPU smoke** và đợi PASS. Không bỏ qua bước này: smoke kiểm tra CUDA, cache và forward của 8 cấu hình, không cập nhật trọng số. Runner cũng thực hiện preflight trước training.
5. Chạy cell điều khiển với `RUN_TRAINING = False`. Đây là xem trước thông số, chưa tạo experiment và chưa train. Kiểm tra LR `0.0003`, reference `exp_002_control_v2`, đủ 8 run, `evaluate_test: False`.

## 3. Chạy thật sau smoke PASS

Cell điều khiển đã được chuẩn bị:

```python
EXPERIMENT_NAME = "exp_003_lr3e4"
EXPERIMENT_CONFIG = "configs/experiments/lr3e4_v2.json"
RUN_TRAINING = False
```

Chỉ đổi `RUN_TRAINING = True` **trong bản điều khiển của bạn**, rồi chạy lại cell này. Không cần push giá trị True. Không cần chạy 03A/03B hoặc train 04A/04B.

Nếu dùng VS Code local + kernel Colab, chỉnh notebook local; checkout Colab vẫn sạch. Nếu mở notebook ngay trong `/content/edge-ai`, dùng bản sao ngoài checkout trước khi chỉnh, vì runner yêu cầu Git sạch để ghi provenance đúng.

Kết quả mới lưu riêng tại:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_003_lr3e4/
```

Runner lưu config/params thực thi, reference, Git commit, runtime, checkpoints, history, predictions, diagnostics và logs. Đợi `COMPLETE 8/8`. Tên đã tồn tại bị từ chối, kể cả FAILED/INTERRUPTED; nếu cần thử lại, dùng `exp_003_lr3e4_retry01` và đặt cùng tên trong 04C. Đây là lần chạy mới, không phải resume.

## 4. Đọc kết quả trong 04C

Chỉ chạy phần đọc kết quả sau khi exp_003 hoàn tất 8/8. 04C đã chọn `EXPERIMENT_NAME = "exp_003_lr3e4"`. Giữ `REFERENCE_EXPERIMENT_NAME = ""` để tự đọc reference exp_002 từ metadata đã lưu.

Chạy setup của 04C trong kernel Colab rồi chạy các cell đọc/hiển thị. Kiểm tra status COMPLETE, TEST False và cột Same runtime trước khi diễn giải so sánh. Xem:

- Chênh lệch validation Macro-F1 từng model với exp_002, đặc biệt TC-ResNet8 03A.
- Validation loss/F1 có bớt dao động và checkpoint còn bị chọn tại epoch 1 không.
- TRAIN eval so với validation, F1/precision/recall Squamosus và bảng HBN/LT.
- Source F1 là chỉ số phụ; không đổi tiêu chí chọn model sau khi nhìn kết quả.

Muốn xem lại lần cũ, đổi riêng tên trong 04C thành `exp_002_control_v2`; không bật training. Sau khi chạy exp_003, gửi output 04C hoặc lưu notebook để đánh giá trước khi chuyển sang thử nghiệm normalization. Một seed chưa đủ kết luận cải thiện ổn định.
