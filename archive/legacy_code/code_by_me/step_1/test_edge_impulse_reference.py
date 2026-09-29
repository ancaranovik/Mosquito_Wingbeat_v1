# ============================================================
# STEP 1B.8B
# PARITY CHECK:
# OUR MFE IMPLEMENTATION vs EDGE IMPULSE SPEECHPY MFE
# ============================================================

from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy import signal


# ============================================================
# 1. LOAD EDGE IMPULSE SPEECHPY
# ============================================================

edge_impulse_repo = Path(
    "D:/VGU-27/Edge AI Project/"
    "third_party/edgeimpulse-processing-blocks"
)

speechpy_parent = (
    edge_impulse_repo
    / "mfe"
    / "third_party"
)

sys.path.insert(
    0,
    str(speechpy_parent)
)

import speechpy


print("============================================================")
print("EDGE IMPULSE REFERENCE IMPORT")
print("============================================================")

print("SpeechPy import: OK")


# ============================================================
# 2. LOAD SAME AUDIO AS OUR PIPELINE
# ============================================================

file_path = (
    "D:/VGU-27/Edge AI Project/"
    "data/audio/220025.wav"
)

sample_rate, audio = wavfile.read(file_path)


# ------------------------------------------------------------
# PCM24 read by scipy is stored inside int32.
#
# Normalize using 2^31 exactly like our previous test.
# ------------------------------------------------------------

audio_float = (
    audio.astype(np.float32)
    / 2147483648.0
)


# ============================================================
# 3. TAKE SAME 1-SECOND SEGMENT
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
# 4. RESAMPLE TO 16 kHz
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

print(
    "Resampled shape:",
    resampled_segment.shape
)

print(
    "Sample rate:",
    target_sample_rate
)


# ============================================================
# 5. EDGE IMPULSE OFFICIAL PRE-EMPHASIS
# ============================================================
#
# Official Edge Impulse MFE implementation >= 3:
#
# coefficient = 0.98
# shift       = 1
#
# We deliberately use THEIR function here.
# ============================================================

ei_preemphasized = (
    speechpy.processing.preemphasis(
        resampled_segment,
        cof=0.98,
        shift=1
    )
)


print("\n============================================================")
print("EDGE IMPULSE PRE-EMPHASIS")
print("============================================================")

print(
    "Shape:",
    ei_preemphasized.shape
)

print(
    "Peak:",
    np.max(
        np.abs(ei_preemphasized)
    )
)


# ============================================================
# 6. EDGE IMPULSE OFFICIAL MFE
# ============================================================
#
# Match our configuration:
#
# implementation version = 4
# frame length           = 20 ms
# frame stride           = 10 ms
# filters                = 40
# FFT                    = 256
# low frequency          = 0 Hz
# high frequency         = 8000 Hz
#
# use_old_mels=False is used by implementation version 4.
# ============================================================

ei_mfe_energy, \
ei_frame_energy, \
ei_filter_freqs, \
ei_filterbank = speechpy.feature.mfe(

    ei_preemphasized,

    sampling_frequency=target_sample_rate,

    implementation_version=4,

    frame_length=0.020,
    frame_stride=0.010,

    num_filters=40,
    fft_length=256,

    low_frequency=0,
    high_frequency=8000,

    use_old_mels=False
)


print("\n============================================================")
print("EDGE IMPULSE OFFICIAL MFE")
print("============================================================")

print(
    "MFE shape:",
    ei_mfe_energy.shape
)

print(
    "Filterbank shape:",
    ei_filterbank.shape
)

print(
    "Min MFE energy:",
    np.min(ei_mfe_energy)
)

print(
    "Max MFE energy:",
    np.max(ei_mfe_energy)
)


# ============================================================
# 7. OUR IMPLEMENTATION
# ============================================================
#
# Now recreate exactly what we wrote in test_mfe_pipeline.py
# so both implementations run inside the SAME file.
#
# This prevents accidental differences from using different
# audio segments or preprocessing.
# ============================================================


# ------------------------------------------------------------
# OUR PRE-EMPHASIS
# ------------------------------------------------------------

previous_samples = np.roll(
    resampled_segment,
    1
)

our_preemphasized = (
    resampled_segment
    - 0.98 * previous_samples
)


# ============================================================
# OUR FRAMING
# ============================================================

frame_length_samples = 320
frame_stride_samples = 160

num_frames = 1 + (
    len(our_preemphasized)
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
        our_preemphasized[
            start:end
        ]
    )


frames = np.array(frames)


# ============================================================
# OUR FFT + POWER SPECTRUM
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

power_spectrum = (
    np.square(fft_magnitude)
    / fft_length
)


# ============================================================
# OUR MEL FILTERBANK
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


our_filterbank = np.zeros(
    (
        num_filters,
        power_spectrum.shape[1]
    )
)


for i in range(num_filters):

    left = bin_points[i]

    center = bin_points[
        i + 1
    ]

    right = bin_points[
        i + 2
    ]


    if center != left:

        for j in range(
            left,
            center
        ):

            our_filterbank[i, j] = (
                (j - left)
                /
                (center - left)
            )


    if right != center:

        for j in range(
            center,
            right
        ):

            our_filterbank[i, j] = (
                (right - j)
                /
                (right - center)
            )


# ============================================================
# OUR MFE ENERGY
# ============================================================

our_mfe_energy = np.dot(
    power_spectrum,
    our_filterbank.T
)

our_mfe_energy = np.maximum(
    our_mfe_energy,
    1e-30
)


print("\n============================================================")
print("OUR MFE")
print("============================================================")

print(
    "MFE shape:",
    our_mfe_energy.shape
)

print(
    "Min:",
    np.min(our_mfe_energy)
)

print(
    "Max:",
    np.max(our_mfe_energy)
)


# ============================================================
# 8. COMPARE PRE-EMPHASIS
# ============================================================

preemphasis_difference = np.abs(
    our_preemphasized
    - ei_preemphasized
)


print("\n============================================================")
print("PRE-EMPHASIS PARITY")
print("============================================================")

print(
    "Max absolute difference:",
    np.max(
        preemphasis_difference
    )
)

print(
    "Mean absolute difference:",
    np.mean(
        preemphasis_difference
    )
)


# ============================================================
# 9. COMPARE MFE
# ============================================================

print("\n============================================================")
print("MFE PARITY")
print("============================================================")


# First confirm shapes match
print(
    "Our shape:",
    our_mfe_energy.shape
)

print(
    "Edge Impulse shape:",
    ei_mfe_energy.shape
)


if (
    our_mfe_energy.shape
    ==
    ei_mfe_energy.shape
):

    difference = np.abs(
        our_mfe_energy
        - ei_mfe_energy
    )


    print(
        "Max absolute difference:",
        np.max(difference)
    )

    print(
        "Mean absolute difference:",
        np.mean(difference)
    )


    # Relative error:
    #
    # |ours - EI| / |EI|
    #
    # epsilon prevents division by zero.
    epsilon = 1e-30

    relative_error = (
        difference
        /
        (
            np.abs(
                ei_mfe_energy
            )
            + epsilon
        )
    )


    print(
        "Max relative error:",
        np.max(relative_error)
    )

    print(
        "Mean relative error:",
        np.mean(relative_error)
    )


    print(
        "np.allclose:",
        np.allclose(
            our_mfe_energy,
            ei_mfe_energy,
            rtol=1e-5,
            atol=1e-12
        )
    )


else:

    print(
        "ERROR: Shapes do not match."
    )

# ============================================================
# STEP 1B.8C - NORMALIZATION + QUANTIZATION PARITY
# ============================================================

noise_floor_db = -52.0


# ============================================================
# OUR NORMALIZATION
# ============================================================

our_mfe_db = 10 * np.log10(
    np.maximum(
        our_mfe_energy,
        1e-30
    )
)

our_normalized = (
    our_mfe_db
    - noise_floor_db
) / (
    (-noise_floor_db)
    + 12.0
)

our_normalized = np.clip(
    our_normalized,
    0.0,
    1.0
)

our_quantized = (
    np.round(
        our_normalized * 256
    )
    / 256.0
)

our_quantized = np.clip(
    our_quantized,
    0.0,
    1.0
)


# ============================================================
# EDGE IMPULSE-STYLE NORMALIZATION
# ============================================================

ei_mfe_db = 10 * np.log10(
    np.maximum(
        ei_mfe_energy,
        1e-30
    )
)

ei_normalized = (
    ei_mfe_db
    - noise_floor_db
) / (
    (-noise_floor_db)
    + 12.0
)

ei_normalized = np.clip(
    ei_normalized,
    0.0,
    1.0
)

ei_quantized = (
    np.round(
        ei_normalized * 256
    )
    / 256.0
)

ei_quantized = np.clip(
    ei_quantized,
    0.0,
    1.0
)


# ============================================================
# COMPARE NORMALIZED VALUES
# ============================================================

normalization_difference = np.abs(
    our_normalized
    - ei_normalized
)

quantization_difference = np.abs(
    our_quantized
    - ei_quantized
)


print("\n============================================================")
print("NORMALIZATION PARITY")
print("============================================================")

print(
    "Max normalized difference:",
    np.max(normalization_difference)
)

print(
    "Mean normalized difference:",
    np.mean(normalization_difference)
)

print(
    "Normalized allclose:",
    np.allclose(
        our_normalized,
        ei_normalized,
        rtol=1e-6,
        atol=1e-12
    )
)


print("\n============================================================")
print("QUANTIZATION PARITY")
print("============================================================")

print(
    "Max quantized difference:",
    np.max(quantization_difference)
)

print(
    "Mean quantized difference:",
    np.mean(quantization_difference)
)

print(
    "Exact equal:",
    np.array_equal(
        our_quantized,
        ei_quantized
    )
)

print(
    "Number of different bins:",
    np.sum(
        our_quantized
        != ei_quantized
    )
)