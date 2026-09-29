from scipy.io import wavfile
from scipy import signal
import numpy as np


# ============================================================
# STEP 1B.1 - RESAMPLE AUDIO TO 16 kHz
# ============================================================

file_path = "D:/VGU-27/Edge AI Project/data/audio/220025.wav"

# Read WAV
sample_rate, audio = wavfile.read(file_path)

# Normalize scipy int32 representation to float [-1, 1]
audio_float = audio.astype(np.float32) / 2147483648.0


# ------------------------------------------------------------
# Take exactly the same 1-second segment: 10s -> 11s
# ------------------------------------------------------------

start_time = 10
duration = 1

start_sample = int(start_time * sample_rate)
end_sample = int((start_time + duration) * sample_rate)

segment = audio_float[start_sample:end_sample]


print("============================================================")
print("ORIGINAL SEGMENT")
print("============================================================")

print("Sample rate:", sample_rate)
print("Shape:", segment.shape)
print("Duration:", len(segment) / sample_rate, "seconds")
print("Min:", np.min(segment))
print("Max:", np.max(segment))
print("Peak:", np.max(np.abs(segment)))


# ============================================================
# RESAMPLE 44.1 kHz -> 16 kHz
# ============================================================
#
# 44100 -> 16000
#
# gcd(44100, 16000) = 100
#
# therefore:
#
# up   = 160
# down = 441
#
# resample_poly() performs:
#
# upsample -> low-pass filter -> downsample
#
# This is preferable to simply throwing samples away.
# ============================================================

target_sample_rate = 16000

resampled_segment = signal.resample_poly(
    segment,
    up=160,
    down=441
)


print("\n============================================================")
print("RESAMPLED SEGMENT")
print("============================================================")

print("Target sample rate:", target_sample_rate)
print("Shape:", resampled_segment.shape)
print(
    "Duration:",
    len(resampled_segment) / target_sample_rate,
    "seconds"
)
print("Min:", np.min(resampled_segment))
print("Max:", np.max(resampled_segment))
print("Peak:", np.max(np.abs(resampled_segment)))

# ============================================================
# STEP 1B.2 - PRE-EMPHASIS
# ============================================================
#
# Edge Impulse MFE implementation >= 3 uses:
#
# coefficient = 0.98
# shift       = 1
#
# Formula:
#
# y[n] = x[n] - 0.98 * x[n-1]
#
# Purpose:
# reduce dominance of slowly-changing / low-frequency content
# and emphasize faster changes in the waveform.
#
# We reproduce Edge Impulse's Python implementation using
# np.roll(signal, 1).
# ============================================================

pre_emphasis_coefficient = 0.98

previous_samples = np.roll(resampled_segment, 1)

preemphasized_segment = (
    resampled_segment
    - pre_emphasis_coefficient * previous_samples
)


print("\n============================================================")
print("STEP 1B.2 - PRE-EMPHASIS")
print("============================================================")

print("Shape:", preemphasized_segment.shape)
print("Duration:",
      len(preemphasized_segment) / target_sample_rate,
      "seconds")

print("\nBefore pre-emphasis:")
print("Min:", np.min(resampled_segment))
print("Max:", np.max(resampled_segment))
print("Peak:", np.max(np.abs(resampled_segment)))

print("\nAfter pre-emphasis:")
print("Min:", np.min(preemphasized_segment))
print("Max:", np.max(preemphasized_segment))
print("Peak:", np.max(np.abs(preemphasized_segment)))


# ============================================================
# STEP 1B.3 - FRAMING
# ============================================================
#
# MFE phân tích audio theo các frame ngắn.
#
# Edge Impulse default:
#
# frame length = 20 ms
# frame stride = 10 ms
#
# Với sample rate = 16 kHz:
#
# frame length:
# 16000 * 0.020 = 320 samples
#
# frame stride:
# 16000 * 0.010 = 160 samples
#
# Vì stride bằng một nửa frame length,
# hai frame liên tiếp overlap 50%.
# ============================================================

frame_length_seconds = 0.020
frame_stride_seconds = 0.010

frame_length_samples = int(
    frame_length_seconds * target_sample_rate
)

frame_stride_samples = int(
    frame_stride_seconds * target_sample_rate
)


print("\n============================================================")
print("STEP 1B.3 - FRAMING")
print("============================================================")

print("Frame length:", frame_length_samples, "samples")
print("Frame stride:", frame_stride_samples, "samples")


# ------------------------------------------------------------
# Calculate number of complete frames
# ------------------------------------------------------------
#
# Formula:
#
# number_frames =
# 1 + floor(
#     (signal_length - frame_length)
#     / frame_stride
# )
#
# Không zero-pad ở test này.
# ------------------------------------------------------------

num_frames = 1 + (
    len(preemphasized_segment) - frame_length_samples
) // frame_stride_samples

print("Number of frames:", num_frames)


# ------------------------------------------------------------
# Create frame matrix
# ------------------------------------------------------------

frames = []

for i in range(num_frames):

    start = i * frame_stride_samples
    end = start + frame_length_samples

    frame = preemphasized_segment[start:end]

    frames.append(frame)


frames = np.array(frames)


print("Frames shape:", frames.shape)

print("\nFirst frame:")
print("Start sample:", 0)
print("End sample:", frame_length_samples)

print("\nSecond frame:")
print("Start sample:", frame_stride_samples)
print(
    "End sample:",
    frame_stride_samples + frame_length_samples
)


# ============================================================
# STEP 1B.4 - FFT AND POWER SPECTRUM
# ============================================================
#
# FFT converts each short frame from:
#
#     amplitude vs time
#
# into:
#
#     magnitude / energy vs frequency
#
# Edge Impulse MFE default FFT length:
#
#     256 points
#
# At 16 kHz:
#
# frequency resolution =
#     16000 / 256
#     = 62.5 Hz per FFT bin
#
# For real-valued audio, rFFT only keeps:
#
#     0 Hz -> Nyquist frequency (8000 Hz)
#
# Therefore:
#
#     256 / 2 + 1 = 129 frequency bins
# ============================================================

fft_length = 256


# ------------------------------------------------------------
# FFT
# ------------------------------------------------------------

fft_complex = np.fft.rfft(
    frames,
    n=fft_length,
    axis=1
)


# Convert complex FFT values to magnitude
fft_magnitude = np.abs(fft_complex)


print("\n============================================================")
print("STEP 1B.4 - FFT")
print("============================================================")

print("Input frames shape:", frames.shape)
print("FFT shape:", fft_magnitude.shape)


# ------------------------------------------------------------
# FREQUENCY AXIS
# ------------------------------------------------------------

fft_frequencies = np.fft.rfftfreq(
    fft_length,
    d=1.0 / target_sample_rate
)

frequency_resolution = target_sample_rate / fft_length


print("Frequency resolution:", frequency_resolution, "Hz")
print("Lowest frequency:", fft_frequencies[0], "Hz")
print("Highest frequency:", fft_frequencies[-1], "Hz")


# ============================================================
# POWER SPECTRUM
# ============================================================
#
# Edge Impulse / SpeechPy MFE calculates power as:
#
#     power = magnitude^2 / FFT_length
#
# Magnitude tells us how strong a frequency component is.
#
# Power is proportional to the ENERGY of that component.
#
# MFE uses POWER, not the raw FFT magnitude.
# ============================================================

power_spectrum = (
    np.square(fft_magnitude)
    / fft_length
)


print("\nPower spectrum:")
print("Shape:", power_spectrum.shape)
print("Min:", np.min(power_spectrum))
print("Max:", np.max(power_spectrum))


# ------------------------------------------------------------
# Find strongest frequency in first frame
# ------------------------------------------------------------

first_frame_power = power_spectrum[0]

strongest_bin = np.argmax(first_frame_power)
strongest_frequency = fft_frequencies[strongest_bin]


print("\nFirst frame:")
print("Strongest FFT bin:", strongest_bin)
print("Strongest frequency:", strongest_frequency, "Hz")
print(
    "Power at strongest frequency:",
    first_frame_power[strongest_bin]
)

# ============================================================
# STEP 1B.5 - MEL FILTERBANK
# ============================================================
#
# Edge Impulse default MFE:
#
# number of Mel filters = 40
# low frequency         = 0 Hz
# high frequency        = sample_rate / 2 = 8000 Hz
#
# Input:
#     power spectrum: (99, 129)
#
# Output:
#     MFE energies:    (99, 40)
# ============================================================

num_filters = 40
low_frequency = 0
high_frequency = target_sample_rate / 2


# ------------------------------------------------------------
# Convert Hz <-> Mel
# ------------------------------------------------------------

def hz_to_mel(frequency):
    return 2595 * np.log10(
        1 + frequency / 700
    )


def mel_to_hz(mel):
    return 700 * (
        10 ** (mel / 2595) - 1
    )


# ------------------------------------------------------------
# Create 40 Mel filters
# ------------------------------------------------------------

low_mel = hz_to_mel(low_frequency)
high_mel = hz_to_mel(high_frequency)

# Need 40 filters + two boundary points
mel_points = np.linspace(
    low_mel,
    high_mel,
    num_filters + 2
)

hz_points = mel_to_hz(mel_points)


# Convert frequencies to FFT bin indices
bin_points = np.floor(
    (fft_length + 1)
    * hz_points
    / target_sample_rate
).astype(int)


# Filterbank matrix:
#
# 40 filters × 129 FFT bins
mel_filterbank = np.zeros(
    (num_filters, len(fft_frequencies))
)


for i in range(num_filters):

    left = bin_points[i]
    center = bin_points[i + 1]
    right = bin_points[i + 2]

    # Rising side of triangle
    if center != left:
        for j in range(left, center):
            mel_filterbank[i, j] = (
                (j - left)
                / (center - left)
            )

    # Falling side of triangle
    if right != center:
        for j in range(center, right):
            mel_filterbank[i, j] = (
                (right - j)
                / (right - center)
            )


print("\n============================================================")
print("STEP 1B.5 - MEL FILTERBANK")
print("============================================================")

print("Power spectrum shape:", power_spectrum.shape)
print("Mel filterbank shape:", mel_filterbank.shape)


# ============================================================
# APPLY MEL FILTERBANK
# ============================================================
#
# (99 × 129) @ (129 × 40)
#
# =
#
# (99 × 40)
# ============================================================

mfe_energy = np.dot(
    power_spectrum,
    mel_filterbank.T
)


# Avoid exact zeros before log10 later
mfe_energy = np.maximum(
    mfe_energy,
    1e-30
)


print("MFE energy shape:", mfe_energy.shape)

print("Min MFE energy:", np.min(mfe_energy))
print("Max MFE energy:", np.max(mfe_energy))


# ------------------------------------------------------------
# Inspect first frame
# ------------------------------------------------------------

first_mfe_frame = mfe_energy[0]

strongest_mel_filter = np.argmax(first_mfe_frame)

mel_center_frequencies = hz_points[1:-1]

print("\nFirst frame:")
print(
    "Strongest Mel filter:",
    strongest_mel_filter
)

print(
    "Approx. center frequency:",
    mel_center_frequencies[strongest_mel_filter],
    "Hz"
)

print(
    "Energy:",
    first_mfe_frame[strongest_mel_filter]
)


# ============================================================
# STEP 1B.6 - EDGE IMPULSE STYLE MFE NORMALIZATION
# ============================================================
#
# Ta đã có:
#
#     mfe_energy
#
# shape:
#
#     (99 frames, 40 Mel filters)
#
# Bây giờ chuyển ENERGY -> dB:
#
#     dB = 10 * log10(energy)
#
# Lưu ý:
# đây là ENERGY / POWER nên dùng 10*log10,
# KHÔNG phải 20*log10 như magnitude.
# ============================================================

mfe_db = 10 * np.log10(mfe_energy)


print("\n============================================================")
print("STEP 1B.6 - MFE dB")
print("============================================================")

print("MFE dB shape:", mfe_db.shape)
print("Min MFE dB:", np.min(mfe_db))
print("Max MFE dB:", np.max(mfe_db))


# ============================================================
# TEST -52 dB NOISE FLOOR
# ============================================================

noise_floor_db = -52.0

below_noise_floor = mfe_db < noise_floor_db

num_below = np.sum(below_noise_floor)
num_total = mfe_db.size

percentage_below = (
    100 * num_below / num_total
)


print("\nNoise floor test:")

print("Total MFE bins:", num_total)
print(
    "Bins below -52 dB:",
    num_below
)
print(
    "Percentage below -52 dB:",
    percentage_below
)


# ============================================================
# NORMALIZE LIKE EDGE IMPULSE
# ============================================================
#
# For noise floor = -52:
#
# normalized =
#     (mfe_db + 52) / 64
#
# General formula:
#
# (mfe_db - noise_floor_db)
# --------------------------------
# (-noise_floor_db) + 12
#
# Then clip to [0, 1].
# ============================================================

mfe_normalized = (
    mfe_db - noise_floor_db
) / (
    (-noise_floor_db) + 12.0
)

mfe_normalized = np.clip(
    mfe_normalized,
    0.0,
    1.0
)


print("\nAfter Edge Impulse normalization:")

print(
    "Min:",
    np.min(mfe_normalized)
)

print(
    "Max:",
    np.max(mfe_normalized)
)

print(
    "Number of zeros:",
    np.sum(mfe_normalized == 0)
)

print(
    "Percentage zeros:",
    100
    * np.sum(mfe_normalized == 0)
    / mfe_normalized.size
)


# ============================================================
# QUANTIZE / DEQUANTIZE TO 8-BIT-LIKE LEVELS
# ============================================================
#
# Edge Impulse further rounds features
# to 1/256 increments.
#
# Example:
#
# 0.41726
#       ↓
# round(0.41726 * 256) / 256
#       ↓
# 0.41796875
#
# The value remains float afterwards,
# but can only take discrete 1/256 steps.
# ============================================================

mfe_quantized = (
    np.round(mfe_normalized * 256.0)
    / 256.0
)

mfe_quantized = np.clip(
    mfe_quantized,
    0.0,
    1.0
)


print("\nAfter 8-bit-like quantization:")

print(
    "Min:",
    np.min(mfe_quantized)
)

print(
    "Max:",
    np.max(mfe_quantized)
)


# ============================================================
# INSPECT FIRST FRAME
# ============================================================

first_frame_db = mfe_db[0]

first_frame_normalized = mfe_normalized[0]

strongest_filter = np.argmax(
    first_frame_db
)


print("\nFirst frame strongest Mel band:")

print(
    "Filter:",
    strongest_filter
)

print(
    "Center frequency:",
    mel_center_frequencies[strongest_filter],
    "Hz"
)

print(
    "Energy:",
    mfe_energy[0, strongest_filter]
)

print(
    "dB:",
    first_frame_db[strongest_filter]
)

print(
    "Normalized:",
    first_frame_normalized[strongest_filter]
)

print(
    "Quantized:",
    mfe_quantized[0, strongest_filter]
)


# ============================================================
# STEP 1B.7 - VISUALIZE MFE BEFORE / AFTER NORMALIZATION
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Time axis
#
# 99 frames, mỗi frame dịch 10 ms
# ------------------------------------------------------------

mfe_times = np.arange(mfe_db.shape[0]) * frame_stride_seconds


# ============================================================
# BEFORE NOISE FLOOR
# ============================================================

plt.figure(figsize=(10, 5))

plt.imshow(
    mfe_db.T,
    origin="lower",
    aspect="auto",
    extent=[
        mfe_times[0],
        mfe_times[-1],
        mel_center_frequencies[0],
        mel_center_frequencies[-1]
    ]
)

plt.colorbar(label="Mel energy (dB)")

plt.xlabel("Time (s)")
plt.ylabel("Approx. Mel center frequency (Hz)")

plt.title("MFE BEFORE -52 dB noise floor")

plt.tight_layout()
plt.show()


# ============================================================
# AFTER EDGE IMPULSE NORMALIZATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.imshow(
    mfe_quantized.T,
    origin="lower",
    aspect="auto",
    extent=[
        mfe_times[0],
        mfe_times[-1],
        mel_center_frequencies[0],
        mel_center_frequencies[-1]
    ],
    vmin=0,
    vmax=1
)

plt.colorbar(label="Normalized MFE feature")

plt.xlabel("Time (s)")
plt.ylabel("Approx. Mel center frequency (Hz)")

plt.title("MFE AFTER Edge Impulse -52 dB normalization")

plt.tight_layout()
plt.show()