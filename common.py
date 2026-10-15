"""Shared bits used by both train.py and app.py."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "organamnist_64.npz"            # full dataset, needed for training
TEST_PATH = ROOT / "data" / "organamnist_64_test.npz"       # test split only, shipped with the app
ART = ROOT / "artifacts"

# Label order is fixed by the MedMNIST v2 OrganAMNIST release.
CLASSES = [
    "bladder", "femur-left", "femur-right", "heart", "kidney-left",
    "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen",
]

# ImageNet statistics, because the backbone was pretrained on ImageNet.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(split):
    d = np.load(DATA_PATH if DATA_PATH.exists() or split != "test" else TEST_PATH)
    return d[f"{split}_images"], d[f"{split}_labels"].ravel().astype(np.int64)


def to_tensor(images_u8):
    """uint8 (N,64,64) numpy or tensor -> normalised float (N,3,64,64) on the same device."""
    x = torch.as_tensor(images_u8)
    x = x.float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_model(pretrained=True):
    """ResNet18 adapted to 64x64 inputs.

    Pretrained ImageNet weights are kept (transfer learning). We only change the
    stride of the first conv and drop the max-pool so a 64px image still gives an
    8x8 feature map at the last block — enough resolution for Grad-CAM.
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.conv1.stride = (1, 1)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m
