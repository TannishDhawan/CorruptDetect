import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
FAMILIES = ["Clean", "Noise", "Blur", "Weather", "Digital"]


def apply_synthetic_corruption(img_np, corr_type, rng):
    """img_np: uint8 RGB 32x32x3. corr_type: 0=Clean, 1=Noise, 2=Blur, 3=Weather, 4=Digital."""
    if corr_type == 0:
        return img_np
    img = img_np.astype(np.float32)
    if corr_type == 1:    # Gaussian noise
        img = img + rng.normal(0.0, rng.uniform(12.0, 35.0), img.shape)
    elif corr_type == 2:  # Gaussian blur
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), sigmaX=float(rng.uniform(1.0, 2.5)))
    elif corr_type == 3:  # Fog-like brightness/contrast shift
        img = img * 0.75 + rng.uniform(35.0, 75.0)
    elif corr_type == 4:  # Pixelation
        s = int(rng.choice([8, 12, 16]))
        small = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(small, (32, 32), interpolation=cv2.INTER_NEAREST)
    return np.clip(img, 0, 255).astype(np.uint8)


class MultiTaskCIFAR10(Dataset):
    """
    split:   'train' (45k) | 'val' (5k held out from train) | 'test' (official 10k, evaluation only)
    corrupt: True  -> each image gets one of the 5 conditions
             False -> clean images only (baseline training, clean accuracy)
    """
    def __init__(self, root="./data", split="train", corrupt=True):
        assert split in ("train", "val", "test")
        self.base = datasets.CIFAR10(root=root, train=(split != "test"), download=True)
        if split == "test":
            self.indices = np.arange(len(self.base))
        else:
            perm = np.random.RandomState(0).permutation(len(self.base))
            self.indices = perm[:45000] if split == "train" else perm[45000:]
        self.split = split
        self.corrupt = corrupt
        self.aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        pil_img, obj_label = self.base[int(self.indices[i])]
        if self.split == "train":
            pil_img = self.aug(pil_img)
            rng = np.random.default_rng()       # fresh randomness every call
            corr_label = int(rng.integers(0, 5)) if self.corrupt else 0
        else:
            rng = np.random.default_rng(i)      # fixed per image -> repeatable evaluation
            corr_label = i % 5 if self.corrupt else 0
        img = apply_synthetic_corruption(np.array(pil_img), corr_label, rng)
        return self.to_tensor(img), obj_label, corr_label
