# Hướng dẫn chạy 8 cấu hình và quản lý kết quả

**Lần tiếp theo exp_008_power075_wd1e4:** làm theo [hướng dẫn exp008](EXP_008_WEIGHT_DECAY.md),
chỉ đổi Adam weight_decay 0 → 0.0001 so với exp007; giữ exponent 0.75, LR 0.001 và normalization.
Các ví dụ exp_001 / baseline_suite bên dưới mô tả luồng baseline gốc và vẫn được
hỗ trợ; notebook 04D hiện mặc định dùng power075_wd1e4_v5 với TEST tắt khi tuning.

**Một lần chạy 04D = một experiment có tên = 8 cấu hình, chạy lần lượt.**
Không cần train trong 04A/04B rồi mới chạy 04D. Không cần chạy lại từ 03 khi chỉ
cập nhật code hoặc muốn một lượt train mới trên cache đã xác thực.

## Vai trò các notebook

| Notebook | Bạn dùng để làm gì? |
|---|---|
| 00–03B | Khoa học/provenance dữ liệu, split và frontend đã cố định; không phải bước chạy lại mỗi lần update |
| 04A | Kiểm tra cache 03A và protocol lịch sử; training baseline đã khóa |
| 04B | Kiểm tra cache 03B và protocol lịch sử; training baseline đã khóa |
| 04D | Clone/pull GitHub, mount Drive, thiết lập package, xác thực dữ liệu/cache, kiểm tra GPU và chạy experiment mới gồm 8 cấu hình |
| 04C | Đọc baseline hoặc experiment đã hoàn tất; xem bảng 8 cấu hình, history và confusion matrix; không train |

## 1. Đồng bộ code trước khi chạy

Sau khi code thay đổi: commit/push trên máy local, rồi chạy lại cell clone/pull
trong 04D trên Colab. Nếu runtime đã import phiên bản module cũ, restart kernel
và chạy lại các cell thiết lập. Không cần upload lại dữ liệu/cache khi chỉ sửa code.

Giữ `RUN_TRAINING = False` trong notebook được commit. Kiểm tra diff và chỉ stage
file cần thiết, tránh gom mọi output ngoài ý muốn. Với thay đổi full-suite này:

```powershell
Set-Location 'D:\VGU-27\Edge AI Project'
git diff --stat
git add -- tools/experiment_runner.py tools/experiment_review.py tools/test_experiment_suite.py tools/test_portability.py configs/experiments/baseline_suite.json notebooks/current/04C_compare_models.ipynb notebooks/current/04D_experiments.ipynb README.md docs/PORTABLE_WORKFLOW.md docs/HUONG_DAN_CHAY_THU_NGHIEM.md
git diff --cached --stat
git commit -m "Add named eight-configuration experiments and saved-result review"
git push origin main
```

Tên experiment được truyền lúc chạy, không cần tạo/commit config mới chỉ để đổi tên.
Runner yêu cầu checkout runtime sạch để Git commit định danh đúng code thực sự chạy.

## 2. Chạy các bước thiết lập trong 04D

1. Chọn Colab GPU runtime.
2. Chạy mount Drive + clone/pull repo.
3. Chạy `configure(install=IN_COLAB)`.
4. Chạy `verify_portability.py`, yêu cầu PASS.
5. Chạy GPU smoke, yêu cầu PASS và `training_executed: false`.
6. Tới cell có `EXPERIMENT_NAME` và `RUN_TRAINING`.

Code nằm trong checkout GitHub; dữ liệu nằm tại
`/content/drive/MyDrive/EdgeAI`. Baseline/cache/raw nằm dưới `fixed/` và được bảo vệ
bằng guard ghi cùng kiểm tra hash. Không sửa hoặc ghi đè chúng.

## 3. Đặt tên tại MỘT CHỖ trong 04D

Cell điều khiển cuối notebook:

```python
EXPERIMENT_NAME = "exp_001_full_suite"
EXPERIMENT_DESCRIPTION = "Lần 1: đủ 8 cấu hình, protocol CUDA cố định, cache đã xác thực."
RUN_TRAINING = False
```

- Đổi `EXPERIMENT_NAME` thành tên chưa dùng: bắt đầu bằng `exp_`, chỉ chữ, số, `_`, `-`.
- Ghi mục đích và thay đổi so với lần trước trong `EXPERIMENT_DESCRIPTION`.
- Sau khi kiểm tra PASS và bạn muốn train thật, đổi `RUN_TRAINING = True` rồi chạy cell đó.
- Không cần đổi frontend/model trong config; template full-suite đã chứa đủ 8 cặp.

Ví dụ lần sau: `exp_002_full_suite`. Nếu chỉ chạy lại cùng phương pháp, ghi rõ là
repeat, không gọi đó là nâng cấp. Khi có phương pháp mới đã được triển khai, ghi ID
lần trước và thay đổi; protocol/params thực tế cùng Git commit vẫn được lưu tự động.

Nếu bạn mở notebook trên VS Code local nhưng chạy bằng kernel Colab, chỉnh các
biến ở file local không sửa checkout Colab. Nếu mở file notebook ngay trong checkout
runtime, dùng một bản notebook điều khiển ngoài checkout để tránh làm bẩn Git.
Đổi cờ về False trước khi lưu/commit bản dùng lâu dài; không cần push cờ True hoặc
push mỗi lần đặt tên mới.

Một cách chạy từ notebook điều khiển ngoài checkout (sau khi đã thiết lập):

```python
import os, subprocess, sys
from pathlib import Path
repo = Path(os.environ["EDGEAI_REPO_ROOT"])
EXPERIMENT_NAME = "exp_001_full_suite"
subprocess.run([
    sys.executable, "-u", "-B", "tools/experiment_runner.py", "train",
    "--config", "configs/experiments/baseline_suite.json",
    "--experiment-id", EXPERIMENT_NAME,
    "--description", "Lần 1: đủ 8 cấu hình, giữ protocol cố định.",
], cwd=repo, check=True)
```

Đây là lệnh train thật, chỉ chạy khi bạn chủ động muốn bắt đầu.

## 4. Thứ tự và nơi lưu kết quả

| Lượt | Frontend | Model |
|---|---|---|
| 1 | 03A | small_cnn |
| 2 | 03A | ds_cnn |
| 3 | 03A | tc_resnet8 |
| 4 | 03A | cnn_lstm |
| 5 | 03B | small_cnn |
| 6 | 03B | ds_cnn |
| 7 | 03B | tc_resnet8 |
| 8 | 03B | cnn_lstm |

Chạy lần lượt trong một lệnh; mỗi cặp vẫn có model, optimizer, seed và checkpoint
riêng theo protocol hiện có. Không chạy 8 model đồng thời trên GPU.

```text
MyDrive/EdgeAI/experiments/exp_001_full_suite/
├── config/
│   ├── experiment.json       tên, mô tả, 2 frontend, 4 model, seed, baseline cha
│   └── metadata.json         commit, runtime/GPU, hash cache/split, tiến độ/status
├── checkpoints/<run_id>/best_model.pt
├── results/
│   ├── protocol.json         params/phương pháp thực tế của lần chạy
│   ├── runs/<run_id>/        history, selection, result, predictions, checkpoint, confusion
│   ├── comparison.csv        bảng tổng hợp 8 cấu hình
│   └── comparison.json       bảng và shortlist dựa trên validation
└── logs/console.log          tiến độ 1/8–8/8 và log epoch/lỗi
```

`run_id` có dạng `03A_small_cnn_seed42`. Log được hiện trực tiếp và lưu lên Drive.
Bảng tổng hợp chỉ xuất sau khi đủ 8 run hoàn tất và được xác thực. TEST không dùng
để chọn checkpoint hoặc shortlist.

Lần 1 luôn ở `experiments/exp_001_full_suite/`; lần 2 ở
`experiments/exp_002_full_suite/`. Update code không di chuyển kết quả cũ. Tên đã tồn
tại bị từ chối trước khi train. Lỗi giữa chừng giữ kết quả đã có và ghi FAILED;
ngắt bằng KeyboardInterrupt/SystemExit ghi INTERRUPTED. Nếu runtime bị tắt cưỡng
bức, metadata có thể còn RUNNING — không coi đó là hoàn tất. Đọc log, giữ nguyên
thư mục và dùng tên mới cho lần thử lại; chưa hỗ trợ tự resume.

## 5. Xem kết quả bằng 04C

Trong cell chọn kết quả của 04C:

```python
EXPERIMENT_NAME = "exp_001_full_suite"
```

Chạy các cell đọc/hiển thị. 04C xác thực hash, đủ 8 run và status COMPLETE; hiển thị
commit, mô tả, runtime, params, bảng, history và confusion matrix. Nó không train,
không chạy lại TEST và không sửa kết quả. Đặt `EXPERIMENT_NAME = ""` để xem baseline
cố định. Muốn xem lần 2 chỉ đổi tên tương ứng; không bật training trong 04D.

## 6. Khi cần thay đổi phương pháp

Hiện protocol cố định: seed 42, Adam, learning rate 0.001, batch 64, tối đa 50 epoch,
early stopping patience 8, chọn checkpoint theo validation weighted loss.
Không thêm learning rate/loss/augmentation tùy ý vào JSON hoặc sửa global CONFIG
để vượt khóa. Các thay đổi đó cần triển khai và kiểm tra riêng trước khi train.

Nếu chỉ sửa đường dẫn, báo cáo hoặc code không đổi feature: dùng cache hiện có,
không chạy lại 03. Nếu thay audio/split/segmentation/frontend: cần đánh giá và tạo
phiên bản dữ liệu/cache mới dưới `derived/`, không ghi đè `fixed/`.

Khi có lỗi, gửi traceback, tên experiment và `config/metadata.json`/log liên quan.
Tôi có thể sửa code local và chạy kiểm tra không training; bạn commit/push, pull
code mới trong Colab rồi chủ động train với tên mới. Kết quả GPU trên Drive là
bằng chứng của lần thực chạy; kiểm tra local không chứng minh GPU training đã PASS.

## 7. LOCAL và COLAB

LOCAL dùng checkout project hiện tại. Có thể đặt `EDGEAI_DATA_ROOT` đến một EdgeAI
root có cấu trúc tương đương Drive; output vào `<root>/experiments/`. Nếu không đặt,
input dùng bản local hiện có và output vào `_runtime/experiments/`. Local CPU smoke
không train; training vẫn yêu cầu CUDA.

COLAB dùng checkout GitHub, data ở `/content/drive/MyDrive/EdgeAI`, kết quả chỉ ở
`/content/drive/MyDrive/EdgeAI/experiments/<EXPERIMENT_NAME>/`. Không phụ thuộc tên
thư mục `.venv-stage04`. 04A/B/C có bootstrap cho kernel riêng; package environment
vẫn phải được thiết lập đúng trong runtime bạn sử dụng.
