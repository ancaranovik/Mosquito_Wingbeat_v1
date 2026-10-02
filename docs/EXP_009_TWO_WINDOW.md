# exp009 — kiểm tra gộp hai window, không train

Đây là bước đầu đã thống nhất sau review exp008. Dùng xác suất validation của **exp_007_power075 / 03A_ds_cnn_seed42** đã lưu. Không cần GPU, raw audio, features.npy, tạo lại cache, checkpoint inference hoặc optimizer. Chỉ đọc artifact của run đã chọn và manifest nhỏ để kiểm tra UID/nhãn/thời gian. Chưa triển khai augmentation hoặc train lại.

## Bạn thao tác như sau

1. Commit và push các file bên dưới từ máy local. Những output exp008 đang có trong notebook được giữ lại; đã backup notebook trước thay đổi dưới `_local_only/notebook_snapshots/before_exp009_20261002/`. Không stage `_local_only`, data, cache hay checkpoint.
2. Mở **04C_compare_models.ipynb mới nhất trên Colab**, chọn runtime **CPU** (không cần bật GPU).
3. Chạy riêng mục đầu **“exp009 — Gộp hai window từ dự đoán exp007 (CPU, không train)”**. Cell thiết lập mount Drive, clone/pull repo và dùng central path layer. Không cần chạy các mục phía dưới, 04A/04B, cell full audit, GPU smoke hay training của 04D.
4. Để `DIAGNOSTIC_NAME="exp_009_two_window_agg"`, `SAVE_DIAGNOSTIC=False`. Chạy cell thiết lập rồi cell tính để xem preview.
5. Để lưu chính kết quả đó lên Drive, đổi **chỉ `SAVE_DIAGNOSTIC=True`**, chạy lại cell thiết lập và cell tính.
6. Khi muốn xem lại kết quả đã lưu, chạy cell thiết lập và cell **“Đọc lại exp009 đã lưu”**. Giữ nguyên tên lần cũ. Không chạy cell lưu một lần nữa với cùng tên.

Tên đã tồn tại sẽ bị từ chối. Nếu lần lưu thất bại, giữ thư mục để kiểm tra `metadata.json`, rồi dùng tên mới như `exp_009_two_window_agg_retry01`. Không xóa hay ghi đè lần cũ.

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
git diff --check
git add -- tools/bounded_aggregation.py tools/test_bounded_aggregation.py notebooks/current/04C_compare_models.ipynb notebooks/current/04D_experiments.ipynb docs/EXP_009_TWO_WINDOW.md
git diff --cached --stat
git commit -m "Add CPU-only two-window validation diagnostic for exp007"
git push
```

Nếu Colab vẫn báo không tìm thấy `bounded_aggregation`, kiểm tra push đã thành công, restart kernel và chạy lại cell thiết lập. Nếu checkout có thay đổi chưa commit, setup dừng để bạn giữ các thay đổi đó trước khi pull.

## Kết quả nằm ở đâu?

Colab:

```text
/content/drive/MyDrive/EdgeAI/experiments/exp_009_two_window_agg/
  metadata.json          # trạng thái, commit, phiên bản code/runtime, hash output
  result.json            # phương pháp, mốc exp007, hash input, metrics, bootstrap
  summary.csv            # bảng so sánh ba endpoint
  per_class.csv          # precision/recall/F1/support mỗi lớp
  pairs.csv              # hai UID của từng cặp, timestamp, xác suất, dự đoán
  excluded_windows.csv   # 159 window đuôi lẻ trong dữ liệu hiện tại
```

Local dùng `ProjectPaths`: mặc định `<repo>/_runtime/experiments/<tên>/`, hoặc `<EDGEAI_DATA_ROOT>/experiments/<tên>/` nếu cấu hình data root. Local chỉ đọc được exp007 nếu artifact của nó tồn tại đúng tại storage được chọn. Không tự tải dữ liệu từ Drive.

Đây là **diagnostic từ dự đoán đã lưu**, không có tám run training. Vì vậy không đặt `EXPERIMENT_NAME` của các mục training review cũ trong 04C/04D thành exp009; dùng `DIAGNOSTIC_NAME` ở mục mới. Exp007 và exp008 vẫn ở thư mục cũ.

## Phương pháp và cách đọc kết quả

Ghép `(0,1), (2,3), ...` theo `example_index` **trong cùng labeled clip `id`**. Kiểm tra thứ tự, timestamp, source/loài/nhãn và geometry; không ghép qua clip hoặc gap. Trung bình bốn xác suất softmax đã lưu, lấy argmax; khi hòa lấy class index nhỏ nhất. Một window có context 0,975s, stride 0,96s; hai window có 15ms audio chung và tổng support **1,935s**. Các cặp khác nhau không chia sẻ window.

Window cuối của clip có số window lẻ bị loại và ghi vào CSV. Đối chứng một-window cũng loại đúng các mẫu đó, để hai phía sử dụng cùng phần âm thanh giữ lại. Số quyết định vẫn khác: hai quyết định đơn so với một quyết định gộp. Không diễn giải đây là cùng độ trễ hoặc model đơn được nâng cấp.

| Endpoint | Ý nghĩa |
|---|---|
| `all_validation_single_window` | F1 exp007 trên toàn validation, chỉ làm mốc tham khảo |
| `matched_single_window` | F1 từng window trên đúng những window được giữ để ghép |
| `two_window` | F1 dự đoán gộp từng cặp |

So sánh chính là **`two_window − matched_single_window`**, không lấy chênh lệch với toàn validation. Báo cáo kèm per-class và CI bootstrap 4.000 lần, seed 20261002, lấy mẫu theo source có phân tầng lớp và dùng cùng source draw ở cả hai phía. CI chưa bao gồm khác biệt seed training, việc đã chọn checkpoint/tuning validation hoặc phụ thuộc giữa source cùng phiên thu.

Preview trên bản sao artifact đã xác thực tại local:

| Endpoint | Số quyết định | Macro-F1 |
|---|---:|---:|
| Toàn validation, một-window | 4.937 | 60,97% |
| Tập giữ lại, một-window | 4.778 | 61,32% |
| Hai-window | 2.389 | 64,90% |

Giữ đủ 177 source và 269 clip, loại 159 window đuôi lẻ. Chênh lệch **+3,58 điểm**, CI nguồn **[+2,22; +4,85]**. Đây là kết quả phát triển validation với audio dài hơn, chưa phải xác nhận trên TEST. Ngưỡng thực hành đã đề xuất ≥3 điểm chỉ là gợi ý xem xét lợi ích, không bảo đảm ý nghĩa thống kê hay hiệu quả ngoài tập này.

## Chạy bằng terminal

Sau khi cấu hình `EDGEAI_DATA_ROOT` để đọc được exp007:

```text
python -B tools/bounded_aggregation.py
python -B tools/bounded_aggregation.py --save --diagnostic-id exp_009_two_window_agg
python -B tools/bounded_aggregation.py --read-saved --diagnostic-id exp_009_two_window_agg
```

Lệnh đầu chỉ preview. Không chạy lại `experiment_runner.py train`. `RUN_TRAINING` trong 04D đã được đưa về **False**; các output cũ giữ nguyên.

## Kiểm tra trước khi bàn giao

Các tests kiểm tra metric biết trước, population khớp, tail lẻ, không ghép qua clip dù chung source, gap/context/nhãn sai, TEST và UID trùng bị chặn, preview không ghi file, lưu không ghi đè, phát hiện output bị thay đổi và input thay đổi trước khi tạo thư mục. Preview thực tế đã chạy trên bản sao exp007; không ghi lên Drive.

```text
python -B -m unittest discover -s tools -p test_bounded_aggregation.py -v
```
