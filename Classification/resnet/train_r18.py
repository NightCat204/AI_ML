import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DATASET_ROOT = os.path.join(_PROJECT_DIR, "datasets", "5fbdf571c06d3433df85ac65-momodel", "garbage_26x100")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
SAVE_DIR = os.path.join(_SCRIPT_DIR, "results")

IMG_DIM = 224
TRAIN_BATCH = 32
LOADER_WORKERS = 4

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20

LR_PHASE1 = 1e-3
LR_PHASE2 = 3e-5
WEIGHT_DECAY = 1e-4

SEED = 42


def fix_random_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_DIM, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=train_transform)

    merged_ds = ConcatDataset([train_ds, val_ds])

    loader = DataLoader(
        merged_ds,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=LOADER_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_ds, merged_ds, loader


def init_resnet18(num_classes):
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def lock_backbone(net):
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True


def unlock_all_layers(net):
    for p in net.parameters():
        p.requires_grad = True


def run_single_epoch(net, loader, criterion, optimizer, device):
    net.train()
    accum_loss = 0.0
    accum_correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)

        bs = labels.size(0)
        accum_loss += loss.item() * bs
        accum_correct += torch.sum(preds == labels).item()
        total += bs

    return accum_loss / total, accum_correct / total


def main():
    fix_random_state(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    base_ds, merged_ds, loader = prepare_dataloader()

    num_classes = len(base_ds.classes)
    print("Number of classes:", num_classes)
    print("Total train samples:", len(merged_ds))
    print("Classes:", base_ds.classes)

    mapping_path = os.path.join(SAVE_DIR, "label_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(base_ds.class_to_idx, f, ensure_ascii=False, indent=2)
    print("Saved class mapping to:", mapping_path)

    net = init_resnet18(num_classes).to(device)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    # ---- Phase 1: 仅训练 FC 层 ----
    print("\n===== Phase 1: train fc only =====")
    lock_backbone(net)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=LR_PHASE1,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for ep in range(PHASE1_EPOCHS):
        t0 = time.time()
        loss, acc = run_single_epoch(net, loader, criterion, optimizer, device)
        scheduler.step()
        dt = time.time() - t0

        print(
            "Phase1 [{}/{}] | loss: {:.4f}, acc: {:.4f} | {:.1f}s".format(
                ep + 1, PHASE1_EPOCHS, loss, acc, dt
            )
        )

    # ---- Phase 2: 全网络微调 ----
    print("\n===== Phase 2: fine-tune all layers =====")
    unlock_all_layers(net)

    optimizer = optim.Adam(
        net.parameters(),
        lr=LR_PHASE2,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for ep in range(PHASE2_EPOCHS):
        t0 = time.time()
        loss, acc = run_single_epoch(net, loader, criterion, optimizer, device)
        scheduler.step()
        dt = time.time() - t0

        print(
            "Phase2 [{}/{}] | loss: {:.4f}, acc: {:.4f} | {:.1f}s".format(
                ep + 1, PHASE2_EPOCHS, loss, acc, dt
            )
        )

    save_path = os.path.join(SAVE_DIR, "r18_gc26.pth")
    torch.save(net.state_dict(), save_path)
    print("\nModel saved to:", save_path)


if __name__ == "__main__":
    main()
