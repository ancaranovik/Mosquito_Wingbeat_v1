from pathlib import Path
import sys
import importlib.util


# ============================================================
# STEP 1B.8D
# LOAD OFFICIAL EDGE IMPULSE MFE PROCESSING BLOCK
# ============================================================

repo = Path(
    "D:/VGU-27/Edge AI Project/"
    "third_party/edgeimpulse-processing-blocks"
)

mfe_folder = repo / "mfe"

# Edge Impulse dsp.py imports modules relative to mfe/
sys.path.insert(0, str(mfe_folder))


# Load mfe/dsp.py explicitly
dsp_path = mfe_folder / "dsp.py"

spec = importlib.util.spec_from_file_location(
    "edge_impulse_mfe_dsp",
    dsp_path
)

ei_mfe_dsp = importlib.util.module_from_spec(spec)

spec.loader.exec_module(
    ei_mfe_dsp
)


print("============================================================")
print("EDGE IMPULSE FULL MFE BLOCK IMPORT")
print("============================================================")

print("Import: OK")
print(
    "generate_features:",
    ei_mfe_dsp.generate_features
)

# ============================================================
# STEP 1B.8D
# FULL EDGE IMPULSE MFE BLOCK PARITY TEST
# ============================================================

import numpy as np
from scipy.io import wavfile
from scipy import signal


# ============================================================
# 1. LOAD SAME WAV
# ============================================================

file_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/audio/220025.wav"
)

sample_rate, audio = wavfile.read(file_path)


# PCM24 is stored by scipy inside int32.
# Normalize to approximately [-1, 1].
audio_float = (
    audio.astype(np.float32)
    / 2147483648.0
)


# ============================================================
# 2. TAKE SAME 1-SECOND SEGMENT: 10s -> 11s
# ============================================================

start_time = 10
duration = 1

start_sample = int(
    start_time * sample_rate
)

end_sample = int(
    (start_time + duration)
    * sample_rate
)

segment = audio_float[
    start_sample:end_sample
]


# ============================================================
# 3. RESAMPLE 44.1 kHz -> 16 kHz
# ============================================================

target_sample_rate = 16000

resampled_segment = signal.resample_poly(
    segment,
    up=160,
    down=441
)


print("\n============================================================")
print("INPUT")
print("============================================================")

print("Shape:", resampled_segment.shape)
print("Min:", np.min(resampled_segment))
print("Max:", np.max(resampled_segment))
print("Peak:", np.max(np.abs(resampled_segment)))


# ============================================================
# 4. PREPARE INPUT FOR EDGE IMPULSE FULL BLOCK
# ============================================================
#
# IMPORTANT:
#
# Edge Impulse MFE dsp.py implementation >= 3 internally does:
#
#     signal = raw_data / 2**15
#
# Therefore generate_features() expects audio in a
# PCM16-like numerical scale.
#
# Our resampled_segment is already normalized [-1, 1],
# so convert it back:
#
#     [-1, 1] -> approximately [-32768, 32768]
#
# Edge Impulse will then divide by 32768 internally.
# ============================================================

ei_input = (
    resampled_segment.astype(np.float32)
    * 32768.0
)


print("\n============================================================")
print("EDGE IMPULSE INPUT")
print("============================================================")

print("Shape:", ei_input.shape)
print("Min:", np.min(ei_input))
print("Max:", np.max(ei_input))
print("Peak:", np.max(np.abs(ei_input)))


# ============================================================
# 5. RUN OFFICIAL EDGE IMPULSE FULL MFE BLOCK
# ============================================================
#
# Configuration:
#
# implementation version = 4
# sample rate            = 16000 Hz
# frame length           = 20 ms
# frame stride           = 10 ms
# filters                = 40
# FFT                    = 256
# low frequency          = 0 Hz
# high frequency         = 8000 Hz
# noise floor            = -52 dB
#
# draw_graphs=False because we only need feature values.
# ============================================================

ei_result = ei_mfe_dsp.generate_features(
    implementation_version=4,
    draw_graphs=False,
    raw_data=ei_input,
    axes=[0],
    sampling_freq=target_sample_rate,
    frame_length=0.020,
    frame_stride=0.010,
    num_filters=40,
    fft_length=256,
    low_frequency=0,
    high_frequency=8000,
    win_size=101,
    noise_floor_db=-52
)


# Returned features are flattened:
#
# 99 frames × 40 filters
# =
# 3960 features
ei_full_features = np.array(
    ei_result["features"],
    dtype=np.float32
)


print("\n============================================================")
print("OFFICIAL EDGE IMPULSE FULL BLOCK")
print("============================================================")

print(
    "Feature count:",
    ei_full_features.size
)

print(
    "Min:",
    np.min(ei_full_features)
)

print(
    "Max:",
    np.max(ei_full_features)
)

print(
    "Zeros:",
    np.sum(ei_full_features == 0)
)

print(
    "Percentage zeros:",
    100
    * np.sum(ei_full_features == 0)
    / ei_full_features.size
)

# ============================================================
# STEP 1B.8E
# FINAL FULL-BLOCK PARITY CHECK
# ============================================================


# ============================================================
# 1. OUR PRE-EMPHASIS
# ============================================================

previous_samples = np.roll(
    resampled_segment,
    1
)

our_signal = (
    resampled_segment
    - 0.98 * previous_samples
)


# ============================================================
# 2. OUR FRAMING
# ============================================================

frame_length_samples = 320
frame_stride_samples = 160

num_frames = 1 + (
    len(our_signal)
    - frame_length_samples
) // frame_stride_samples


frames = []

for i in range(num_frames):

    start = (
        i
        * frame_stride_samples
    )

    end = (
        start
        + frame_length_samples
    )

    frames.append(
        our_signal[start:end]
    )


frames = np.array(frames)


# ============================================================
# 3. FFT
# ============================================================

fft_length = 256

fft_complex = np.fft.rfft(
    frames,
    n=fft_length,
    axis=1
)

fft_magnitude = np.abs(
    fft_complex
)


# ============================================================
# 4. POWER SPECTRUM
# ============================================================

power_spectrum = (
    np.square(fft_magnitude)
    / fft_length
)


# ============================================================
# 5. MEL FILTERBANK
# ============================================================

num_filters = 40

low_frequency = 0
high_frequency = 8000


def hz_to_mel(frequency):

    return (
        2595
        * np.log10(
            1
            + frequency / 700
        )
    )


def mel_to_hz(mel):

    return (
        700
        * (
            10 ** (
                mel / 2595
            )
            - 1
        )
    )


low_mel = hz_to_mel(
    low_frequency
)

high_mel = hz_to_mel(
    high_frequency
)


mel_points = np.linspace(
    low_mel,
    high_mel,
    num_filters + 2
)

hz_points = mel_to_hz(
    mel_points
)


bin_points = np.floor(
    (fft_length + 1)
    * hz_points
    / target_sample_rate
).astype(int)


mel_filterbank = np.zeros(
    (
        num_filters,
        power_spectrum.shape[1]
    )
)


for i in range(num_filters):

    left = bin_points[i]
    center = bin_points[i + 1]
    right = bin_points[i + 2]


    if center != left:

        for j in range(
            left,
            center
        ):

            mel_filterbank[i, j] = (
                (j - left)
                /
                (center - left)
            )


    if right != center:

        for j in range(
            center,
            right
        ):

            mel_filterbank[i, j] = (
                (right - j)
                /
                (right - center)
            )


# ============================================================
# 6. MEL ENERGY
# ============================================================

our_mfe = np.dot(
    power_spectrum,
    mel_filterbank.T
)

our_mfe = np.clip(
    our_mfe,
    1e-30,
    None
)


# ============================================================
# 7. dB
# ============================================================

our_mfe = (
    10
    * np.log10(
        our_mfe
    )
)


# ============================================================
# 8. -52 dB NORMALIZATION
# ============================================================

noise_floor_db = -52.0

our_mfe = (
    our_mfe
    - noise_floor_db
) / (
    (-noise_floor_db)
    + 12
)

our_mfe = np.clip(
    our_mfe,
    0,
    1
)


# Count zeros BEFORE quantization
zeros_before_quantization = np.sum(
    our_mfe == 0
)


# ============================================================
# 9. EDGE IMPULSE 8-BIT QUANTIZATION
# ============================================================

our_mfe = np.uint8(
    np.around(
        our_mfe * 256
    )
)

our_mfe = np.clip(
    our_mfe,
    0,
    255
)

our_mfe = np.float32(
    our_mfe / 256
)


# Flatten 99 x 40 -> 3960
our_full_features = (
    our_mfe.flatten()
)


# ============================================================
# 10. FINAL COMPARISON
# ============================================================

difference = np.abs(
    our_full_features
    - ei_full_features
)


print("\n============================================================")
print("FINAL FULL PIPELINE PARITY")
print("============================================================")

print(
    "Our feature count:",
    our_full_features.size
)

print(
    "EI feature count:",
    ei_full_features.size
)

print(
    "Zeros before quantization:",
    zeros_before_quantization
)

print(
    "Zeros after quantization:",
    np.sum(
        our_full_features == 0
    )
)

print(
    "Max absolute difference:",
    np.max(difference)
)

print(
    "Mean absolute difference:",
    np.mean(difference)
)

print(
    "Exact equal:",
    np.array_equal(
        our_full_features,
        ei_full_features
    )
)

print(
    "Number of different features:",
    np.sum(
        our_full_features
        != ei_full_features
    )
)