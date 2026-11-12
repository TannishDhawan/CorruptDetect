import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model import CorruptDetectNet
from dataset import MultiTaskCIFAR10, FAMILIES


def predict(model, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    obj_p, obj_t, corr_p, corr_t = [], [], [], []
    with torch.no_grad():
        for imgs, obj_y, corr_y in loader:
            obj_logits, corr_logits = model(imgs.to(device))
            obj_p.append(obj_logits.argmax(1).cpu().numpy())
            corr_p.append(corr_logits.argmax(1).cpu().numpy())
            obj_t.append(obj_y.numpy())
            corr_t.append(corr_y.numpy())
    return tuple(np.concatenate(x) for x in (obj_p, obj_t, corr_p, corr_t))


def evaluate(ckpt: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptDetectNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Official CIFAR-10 test set: used only here, never during training or checkpoint selection
    obj_p, obj_t, _, _ = predict(model, MultiTaskCIFAR10(split="test", corrupt=False), device)
    clean_acc = 100.0 * (obj_p == obj_t).mean()

    obj_p, obj_t, corr_p, corr_t = predict(model, MultiTaskCIFAR10(split="test", corrupt=True), device)
    corrupted = corr_t != 0

    print(f"\n=== {ckpt} ===")
    print(f"Clean test accuracy (10,000 images)    : {clean_acc:.2f}%")
    print(f"Corrupted test accuracy (8,000 images) : {100.0 * (obj_p[corrupted] == obj_t[corrupted]).mean():.2f}%")
    for k in range(1, 5):
        m = corr_t == k
        print(f"    under {FAMILIES[k]:<8}: {100.0 * (obj_p[m] == obj_t[m]).mean():.2f}%")
    print(f"Corruption detection accuracy          : {100.0 * (corr_p == corr_t).mean():.2f}%")
    print(f"Corruption detection macro-F1          : {100.0 * f1_score(corr_t, corr_p, average='macro', zero_division=0):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    evaluate(parser.parse_args().ckpt)
