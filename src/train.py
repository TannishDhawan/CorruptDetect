import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CorruptDetectNet
from dataset import MultiTaskCIFAR10


def validate(model, loader, device):
    model.eval()
    obj_correct = corr_correct = total = 0
    with torch.no_grad():
        for imgs, obj_y, corr_y in loader:
            imgs, obj_y, corr_y = imgs.to(device), obj_y.to(device), corr_y.to(device)
            obj_logits, corr_logits = model(imgs)
            obj_correct += (obj_logits.argmax(1) == obj_y).sum().item()
            corr_correct += (corr_logits.argmax(1) == corr_y).sum().item()
            total += obj_y.size(0)
    return 100.0 * obj_correct / total, 100.0 * corr_correct / total


def train(mode: str, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrupt = mode == "robust"   # baseline = clean images only
    print(f"Mode: {mode} | Device: {device} | Epochs: {epochs}")

    train_ds = MultiTaskCIFAR10(split="train", corrupt=corrupt)
    val_ds = MultiTaskCIFAR10(split="val", corrupt=corrupt)  # held out from train; test set is not used here
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = CorruptDetectNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs("./models", exist_ok=True)
    ckpt_path = f"./models/{mode}_best.pth"
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, obj_y, corr_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, obj_y, corr_y = imgs.to(device), obj_y.to(device), corr_y.to(device)
            optimizer.zero_grad()
            obj_logits, corr_logits = model(imgs)
            loss = criterion(obj_logits, obj_y) + 0.6 * criterion(corr_logits, corr_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        obj_acc, corr_acc = validate(model, val_loader, device)
        line = f"Epoch {epoch:02d}: loss={total_loss / len(train_loader):.4f} | val object acc={obj_acc:.2f}%"
        if corrupt:
            line += f" | val corruption-detect acc={corr_acc:.2f}%"
        print(line)

        if obj_acc > best_acc:
            best_acc = obj_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nDone. Best val object accuracy {best_acc:.2f}% -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "robust"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.mode, args.epochs)
