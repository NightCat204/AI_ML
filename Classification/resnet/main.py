import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, models


PATH_NET18 = "./results/r18_gc26.pth"
PATH_NET34 = "./results/r34_gc26.pth"
IMG_DIM = 224

idx_to_label = {
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

_hw = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_net18 = None
_net34 = None

_img_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_DIM, IMG_DIM)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _make_net18():
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 26)
    return net


def _make_net34():
    net = models.resnet34(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 26)
    return net


def _ensure_models():
    global _net18, _net34

    if _net18 is not None and _net34 is not None:
        return

    n18 = _make_net18()
    w18 = torch.load(PATH_NET18, map_location=_hw)
    n18.load_state_dict(w18)
    n18.to(_hw)
    n18.eval()

    n34 = _make_net34()
    w34 = torch.load(PATH_NET34, map_location=_hw)
    n34.load_state_dict(w34)
    n34.to(_hw)
    n34.eval()

    _net18 = n18
    _net34 = n34


def _preprocess(image):
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

    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    return image


def predict(image):
    _ensure_models()
    image = _preprocess(image)

    img_flipped = np.ascontiguousarray(image[:, ::-1, :])

    t_orig = _img_pipeline(image).unsqueeze(0).to(_hw)
    t_flip = _img_pipeline(img_flipped).unsqueeze(0).to(_hw)

    with torch.no_grad():
        logits_18 = (_net18(t_orig) + _net18(t_flip)) / 2.0
        logits_34 = (_net34(t_orig) + _net34(t_flip)) / 2.0
        combined = (logits_18 + logits_34) / 2.0
        pred_idx = int(torch.argmax(combined, dim=1).item())

    return idx_to_label[pred_idx]
