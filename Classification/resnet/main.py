import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, models


MODEL18_PATH = "./results/resnet18_final_full.pth"
MODEL34_PATH = "./results/resnet34_final_full.pth"
IMAGE_SIZE = 224

inverted = {
    0: 'Plastic Bottle',
    1: 'Hats',
    2: 'Newspaper',
    3: 'Cans',
    4: 'Glassware',
    5: 'Glass Bottle',
    6: 'Cardboard',
    7: 'Basketball',
    8: 'Paper',
    9: 'Metalware',
    10: 'Disposable Chopsticks',
    11: 'Lighter',
    12: 'Broom',
    13: 'Old Mirror',
    14: 'Toothbrush',
    15: 'Dirty Cloth',
    16: 'Seashell',
    17: 'Ceramic Bowl',
    18: 'Paint bucket',
    19: 'Battery',
    20: 'Fluorescent lamp',
    21: 'Tablet capsules',
    22: 'Orange Peel',
    23: 'Vegetable Leaf',
    24: 'Eggshell',
    25: 'Banana Peel'
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model18 = None
_model34 = None

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _build_resnet18():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 26)
    return model


def _build_resnet34():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 26)
    return model


def _load_once():
    global _model18, _model34

    if _model18 is not None and _model34 is not None:
        return

    model18 = _build_resnet18()
    state18 = torch.load(MODEL18_PATH, map_location=_device)
    model18.load_state_dict(state18)
    model18.to(_device)
    model18.eval()

    model34 = _build_resnet34()
    state34 = torch.load(MODEL34_PATH, map_location=_device)
    model34.load_state_dict(state34)
    model34.to(_device)
    model34.eval()

    _model18 = model18
    _model34 = model34


def _prepare(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim != 3:
        raise ValueError("Input image must have shape (H, W, C)")

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] != 3:
        raise ValueError("Input image channel must be 1 or 3")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image


def predict(image):
    _load_once()
    image = _prepare(image)

    image_flip = np.ascontiguousarray(image[:, ::-1, :])

    tensor1 = _transform(image).unsqueeze(0).to(_device)
    tensor2 = _transform(image_flip).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits18 = (_model18(tensor1) + _model18(tensor2)) / 2.0
        logits34 = (_model34(tensor1) + _model34(tensor2)) / 2.0
        logits = (logits18 + logits34) / 2.0
        pred_idx = int(torch.argmax(logits, dim=1).item())

    return inverted[pred_idx]