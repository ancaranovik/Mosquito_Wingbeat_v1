# ============================================================
# STEP 1A - PREPROCESSING SANITY CHECK
# Mục tiêu:
# 1. Kiểm tra WAV đang được lưu ở format nào
# 2. Xác định amplitude scale của dữ liệu
# 3. Normalize waveform về float [-1, 1]
# 4. Lấy cùng một đoạn audio 1 giây
# 5. So sánh spectrogram của raw scale và normalized scale
# 6. Kiểm tra tác động của floor = -52 dB
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

from scipy.io import wavfile
from scipy import signal
import numpy as np
import wave
import matplotlib.pyplot as plt


# ============================================================
# 1A.1 - READ WAV FILE
# ============================================================

file_path = "D:/VGU-27/Edge AI Project/data/audio/220025.wav"

# scipy.io.wavfile.read() trả:
# - sample_rate: số sample mỗi giây
# - audio: toàn bộ waveform
sample_rate, audio = wavfile.read(file_path)


print("============================================================")
print("STEP 1A.1 - RAW WAV INFORMATION")
print("============================================================")

print("Sample rate:", sample_rate)
print("dtype:", audio.dtype)
print("shape:", audio.shape)

print("min:", np.min(audio))
print("max:", np.max(audio))
print("abs peak:", np.max(np.abs(audio)))


# ============================================================
# Kiểm tra format thật bên trong WAV
# ============================================================
#
# scipy trả dtype=int32 cho PCM24,
# nên chỉ nhìn dtype chưa đủ để biết file là 24-bit hay 32-bit.
#
# wave.getsampwidth():
# 2 bytes -> 16-bit
# 3 bytes -> 24-bit
# 4 bytes -> 32-bit
# ============================================================

with wave.open(file_path, "rb") as wf:

    print("\nWAV format information:")

    print("Channels:", wf.getnchannels())
    print("Sample width:", wf.getsampwidth(), "bytes")
    print("Frame rate:", wf.getframerate())
    print("Frames:", wf.getnframes())


# ============================================================
# 1A.2 - NORMALIZE AUDIO
# ============================================================
#
# File này là PCM24 nhưng scipy lưu nó trong int32 container.
#
# Với scipy, PCM24 được left-justified trong int32.
#
# Do đó để đưa waveform về approximately [-1, +1],
# ta chia cho:
#
# 2^31 = 2147483648
#
# Quan trọng:
# normalize KHÔNG thay đổi âm thanh.
# Nó chỉ thay đổi scale biểu diễn.
#
# Ví dụ:
# raw      = 48,627,712
# normalized = 0.02264
#
# vẫn là cùng một amplitude.
# ============================================================

audio_float = audio.astype(np.float32) / 2147483648.0


print("\n============================================================")
print("STEP 1A.2 - NORMALIZED AUDIO")
print("============================================================")

print("dtype:", audio_float.dtype)
print("min:", np.min(audio_float))
print("max:", np.max(audio_float))
print("abs peak:", np.max(np.abs(audio_float)))


# ============================================================
# 1A.3 - TAKE EXACTLY THE SAME 1-SECOND SEGMENT
# ============================================================
#
# Chọn đoạn từ 10s -> 11s.
#
# Vì sample_rate = 44100 Hz:
#
# 1 second = 44100 samples
#
# segment_raw:
#     waveform ở raw scipy scale
#
# segment_float:
#     cùng chính xác waveform đó
#     nhưng normalized về [-1, 1]
#
# Việc dùng cùng segment rất quan trọng,
# vì ta muốn so sánh scale chứ không phải so hai audio khác nhau.
# ============================================================

start_time = 10      # seconds
duration = 1         # seconds

start_sample = int(start_time * sample_rate)
end_sample = int((start_time + duration) * sample_rate)

segment_raw = audio[start_sample:end_sample]
segment_float = audio_float[start_sample:end_sample]


print("\n============================================================")
print("STEP 1A.3 - 1-SECOND SEGMENT")
print("============================================================")

print("Raw shape:", segment_raw.shape)
print("Float shape:", segment_float.shape)

print("\nRaw segment:")
print("min:", np.min(segment_raw))
print("max:", np.max(segment_raw))
print("peak:", np.max(np.abs(segment_raw)))

print("\nNormalized segment:")
print("min:", np.min(segment_float))
print("max:", np.max(segment_float))
print("peak:", np.max(np.abs(segment_float)))


# ============================================================
# 1A.4 - CREATE NORMALIZED SPECTROGRAM
# ============================================================
#
# Spectrogram cho biết:
#
# frequency thay đổi theo time như thế nào.
#
# nperseg = 1024:
# mỗi FFT xử lý 1024 samples.
#
# noverlap = 512:
# hai window liên tiếp overlap 50%.
#
# mode="magnitude":
# output là magnitude spectrum,
# nên khi đổi sang dB ta dùng:
#
# dB = 20 * log10(magnitude)
#
# Đây CHƯA phải preprocessing tối ưu.
# Mục tiêu hiện tại chỉ là kiểm tra amplitude convention.
# ============================================================

frequencies, times, Sxx_float = signal.spectrogram(
    segment_float,
    fs=sample_rate,
    nperseg=1024,
    noverlap=512,
    scaling="spectrum",
    mode="magnitude"
)


print("\n============================================================")
print("STEP 1A.4 - NORMALIZED SPECTROGRAM")
print("============================================================")

print("Shape:", Sxx_float.shape)
print("Min magnitude:", np.min(Sxx_float))
print("Max magnitude:", np.max(Sxx_float))


# tránh log10(0)
epsilon = 1e-12

Sxx_float_db = 20 * np.log10(Sxx_float + epsilon)


print("Min dB:", np.min(Sxx_float_db))
print("Max dB:", np.max(Sxx_float_db))


# ============================================================
# 1A.5 - CREATE RAW-SCALE SPECTROGRAM
# ============================================================
#
# Dùng CHÍNH XÁC parameters như normalized spectrogram.
#
# Điểm duy nhất khác:
#
# input = segment_raw
#
# Mục tiêu:
# kiểm tra cùng một audio nhưng scale khác nhau
# sẽ làm dB spectrum dịch bao nhiêu.
# ============================================================

frequencies_raw, times_raw, Sxx_raw = signal.spectrogram(
    segment_raw,
    fs=sample_rate,
    nperseg=1024,
    noverlap=512,
    scaling="spectrum",
    mode="magnitude"
)

Sxx_raw_db = 20 * np.log10(Sxx_raw + epsilon)


print("\n============================================================")
print("STEP 1A.5 - RAW SPECTROGRAM")
print("============================================================")

print("Shape:", Sxx_raw.shape)
print("Min magnitude:", np.min(Sxx_raw))
print("Max magnitude:", np.max(Sxx_raw))
print("Min dB:", np.min(Sxx_raw_db))
print("Max dB:", np.max(Sxx_raw_db))


# ============================================================
# CALCULATE dB OFFSET BETWEEN RAW AND NORMALIZED
# ============================================================
#
# Vì:
#
# raw ≈ normalized * 2^31
#
# nên theoretical dB difference:
#
# 20 * log10(2^31)
# ≈ 186.64 dB
#
# Ta kiểm tra thực nghiệm xem có đúng hay không.
# ============================================================

max_db_difference = np.max(Sxx_raw_db) - np.max(Sxx_float_db)
min_db_difference = np.min(Sxx_raw_db) - np.min(Sxx_float_db)


print("\n============================================================")
print("RAW vs NORMALIZED dB DIFFERENCE")
print("============================================================")

print("Max dB difference:", max_db_difference)
print("Min dB difference:", min_db_difference)


# ============================================================
# 1A.6 - TEST -52 dB FLOOR
# ============================================================
#
# floor_db = -52 dB
#
# Ta muốn biết:
# bao nhiêu spectrogram bins đang nằm dưới -52 dB?
#
# Example:
#
# -80 dB  -> below floor
# -60 dB  -> below floor
# -50 dB  -> above floor
#
# Đây chưa kết luận threshold đúng hay sai.
# Nó chỉ cho biết threshold đang tác động đến bao nhiêu dữ liệu.
# ============================================================

floor_db = -52.0

below_floor = Sxx_float_db < floor_db

num_below = np.sum(below_floor)
num_total = Sxx_float_db.size

percentage_below = 100 * num_below / num_total


print("\n============================================================")
print("STEP 1A.6 - -52 dB FLOOR TEST")
print("============================================================")

print("Total bins:", num_total)
print("Bins below -52 dB:", num_below)
print("Percentage below floor:", percentage_below)


# ============================================================
# APPLY FLOOR
# ============================================================
#
# np.maximum(x, -52)
#
# Example:
#
# BEFORE:
# -100
# -80
# -60
# -50
#
# AFTER:
# -52
# -52
# -52
# -50
#
# Floor không xóa bins.
#
# Nó clamp tất cả giá trị nhỏ hơn -52
# thành đúng -52.
# ============================================================

Sxx_floored_db = np.maximum(
    Sxx_float_db,
    floor_db
)


print("\nFloor result:")

print("Before floor min:", np.min(Sxx_float_db))
print("Before floor max:", np.max(Sxx_float_db))

print("After floor min:", np.min(Sxx_floored_db))
print("After floor max:", np.max(Sxx_floored_db))


# ============================================================
# 1A.7 - VISUALIZE SPECTROGRAM BEFORE FLOOR
# ============================================================
#
# Chỉ hiển thị 0-3000 Hz để dễ quan sát
# mosquito wingbeat + harmonics.
#
# vmin / vmax được giữ giống nhau cho cả BEFORE và AFTER
# để comparison công bằng.
# ============================================================

plt.figure(figsize=(10, 5))

plt.pcolormesh(
    times,
    frequencies,
    Sxx_float_db,
    shading="auto",
    vmin=-100,
    vmax=-40
)

plt.colorbar(label="Magnitude (dB)")

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title("Spectrogram BEFORE -52 dB floor")

plt.ylim(0, 3000)

plt.tight_layout()
plt.show()


# ============================================================
# 1A.8 - VISUALIZE SPECTROGRAM AFTER FLOOR
# ============================================================

plt.figure(figsize=(10, 5))

plt.pcolormesh(
    times,
    frequencies,
    Sxx_floored_db,
    shading="auto",
    vmin=-100,
    vmax=-40
)

plt.colorbar(label="Magnitude (dB)")

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title("Spectrogram AFTER -52 dB floor")

plt.ylim(0, 3000)

plt.tight_layout()
plt.show()