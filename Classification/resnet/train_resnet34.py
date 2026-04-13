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


# =========================
# 基本配置
# =========================
DATA_ROOT = "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
RESULTS_DIR = "./results"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20

LR_STAGE1 = 1e-3
LR_STAGE2 = 3e-5
WEIGHT_DECAY = 1e-4

SEED = 42


# =========================
# 随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 数据
# =========================
def build_full_train_loader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
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

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=train_transform)

    full_train_dataset = ConcatDataset([train_dataset, val_dataset])

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataset, full_train_dataset, train_loader


# =========================
# 模型
# =========================
def build_model(num_classes):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# =========================
# 训练一个 epoch
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += torch.sum(preds == labels).item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


# =========================
# 主流程
# =========================
def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    base_train_dataset, full_train_dataset, train_loader = build_full_train_loader()

    num_classes = len(base_train_dataset.classes)
    print("Number of classes:", num_classes)
    print("Full train samples:", len(full_train_dataset))
    print("Classes:", base_train_dataset.classes)

    # 保存类别映射
    class_to_idx_path = os.path.join(RESULTS_DIR, "class_to_idx.json")
    with open(class_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(base_train_dataset.class_to_idx, f, ensure_ascii=False, indent=2)
    print("Saved class_to_idx to:", class_to_idx_path)

    model = build_model(num_classes).to(device)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Stage 1: 只训练最后一层
    # -------------------------
    print("\n===== Stage 1: train fc only on full data =====")
    freeze_backbone(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

    for epoch in range(STAGE1_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(
            "Stage1 Epoch [{}/{}] | train_loss: {:.4f}, train_acc: {:.4f} | time: {:.1f}s".format(
                epoch + 1, STAGE1_EPOCHS, train_loss, train_acc, elapsed
            )
        )

    # -------------------------
    # Stage 2: 解冻全网络微调
    # -------------------------
    print("\n===== Stage 2: fine-tune all layers on full data =====")
    unfreeze_all(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

    for epoch in range(STAGE2_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(
            "Stage2 Epoch [{}/{}] | train_loss: {:.4f}, train_acc: {:.4f} | time: {:.1f}s".format(
                epoch + 1, STAGE2_EPOCHS, train_loss, train_acc, elapsed
            )
        )

    final_path = os.path.join(RESULTS_DIR, "resnet34_final_full.pth")
    torch.save(model.state_dict(), final_path)
    print("\nSaved final model to:", final_path)


if __name__ == "__main__":
    main()