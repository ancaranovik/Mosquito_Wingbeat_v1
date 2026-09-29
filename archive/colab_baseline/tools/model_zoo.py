"""Four fixed, from-scratch Stage-04 models. Input: [batch, 1, time, frequency]."""
import copy

import torch
from torch import nn

FAMILIES = ("small_cnn", "ds_cnn", "tc_resnet8", "cnn_lstm")
DESIGNS = {
    "small_cnn": {"channels": [16, 32, 64], "kernel": [3, 3],
                  "stride": [1, 1], "max_pool_after_first_two": [2, 2],
                  "head": "global_mean_time_frequency -> dropout -> linear4"},
    "ds_cnn": {"stem_channels": 32, "stem_kernel": [3, 3], "stem_stride": [2, 2],
               "blocks": 4, "channels": 32, "depthwise_kernel": [3, 3],
               "block_stride": [1, 1], "pointwise_kernel": [1, 1],
               "head": "global_mean_time_frequency -> dropout -> linear4"},
    "tc_resnet8": {"input_layout": "frequency_as_channels; temporal Conv1d",
                   "channels": [16, 24, 32, 48], "stem_kernel": 3, "stem_stride": 1,
                   "residual_blocks": 3, "convolutions_per_block": 2,
                   "block_kernel": 9, "first_conv_strides": [2, 2, 2],
                   "second_conv_stride": 1, "shortcut": "1x1 projection + BN",
                   "head": "global_mean_time -> dropout -> linear4",
                   "depth_convention": "7 main-path convolutions + classifier; projections excluded"},
    "cnn_lstm": {"channels": [16, 32], "kernel": [3, 3], "stride": [1, 1],
                 "max_pool_after_each": [2, 2], "frequency_reduction": "mean",
                 "lstm_input": 32, "lstm_hidden": 32, "lstm_layers": 1,
                 "bidirectional": False, "lstm_dropout": 0.0,
                 "head": "last temporal hidden state -> dropout -> linear4"},
}


def architecture_config(family, frequency_bins):
    if family not in FAMILIES or frequency_bins not in (40, 64):
        raise ValueError((family, frequency_bins))
    return dict(copy.deepcopy(DESIGNS[family]), family=family, input_shape=[1, 96, frequency_bins],
                classes=4, activation="ReLU", dropout=0.2,
                convolution_bias=False, convolution_padding="symmetric k//2",
                batch_norm={"eps": 1e-5, "momentum": 0.1, "affine": True},
                linear_bias=True, initialization="PyTorch default reset_parameters; seed 42",
                pretrained=False)


def conv2(cin, cout, stride=1, groups=1, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv2(1, 16), nn.MaxPool2d(2), conv2(16, 32),
                                      nn.MaxPool2d(2), conv2(32, 64))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 4))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [conv2(1, 32, stride=2)]
        for _ in range(4):
            layers.extend([conv2(32, 32, groups=32), conv2(32, 32, kernel=1)])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 4))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(cin, cout, 9, 2, 4, bias=False),
                                  nn.BatchNorm1d(cout), nn.ReLU(),
                                  nn.Conv1d(cout, cout, 9, 1, 4, bias=False),
                                  nn.BatchNorm1d(cout))
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, 2, bias=False), nn.BatchNorm1d(cout))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class TCResNet8(nn.Module):
    """TC-ResNet8 adaptation: time=96, F input channels; no image-axis convolution."""
    def __init__(self, frequency_bins):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(frequency_bins, 16, 3, 1, 1, bias=False),
                                      nn.BatchNorm1d(16), nn.ReLU(),
                                      TemporalBlock(16, 24), TemporalBlock(24, 32),
                                      TemporalBlock(32, 48))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(48, 4))

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        return self.head(self.features(x).mean(dim=2))


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv2(1, 16), nn.MaxPool2d(2),
                                      conv2(16, 32), nn.MaxPool2d(2))
        self.lstm = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 4))

    def forward(self, x):
        x = self.features(x).mean(dim=3).transpose(1, 2)  # [N, 24 time steps, 32]
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def build_model(family, frequency_bins):
    architecture_config(family, frequency_bins)  # validate before construction
    if family == "tc_resnet8":
        return TCResNet8(frequency_bins)
    return {"small_cnn": SmallCNN, "ds_cnn": DSCNN, "cnn_lstm": CNNLSTM}[family]()
