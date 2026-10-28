import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision import transforms
import matplotlib.pyplot as plt

register_heif_opener()

class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(self.flatten(x))

model = DigitMLP()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

input_folder = "my_digits"
output_folder = "mnist_format_digits"
os.makedirs(output_folder, exist_ok=True)

files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))])

images_to_show = []

for file_name in files:
    path = os.path.join(input_folder, file_name)
    pil_img = Image.open(path).convert('L')
    img_np = np.array(pil_img)

    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    coords = cv2.findNonZero(dilated)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = dilated[y:y+h, x:x+w]
    else:
        cropped = dilated

    h_c, w_c = cropped.shape
    max_dim = max(h_c, w_c)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    dx = (max_dim - w_c) // 2
    dy = (max_dim - h_c) // 2
    square[dy:dy+h_c, dx:dx+w_c] = cropped

    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    final_28x28 = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    out_name = os.path.splitext(file_name)[0] + "_28x28.png"
    cv2.imwrite(os.path.join(output_folder, out_name), final_28x28)

    tensor_img = torch.tensor(final_28x28, dtype=torch.float32) / 255.0
    tensor_img = transforms.Normalize((0.1307,), (0.3081,))(tensor_img.unsqueeze(0)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_img)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item() * 100

    images_to_show.append((final_28x28, pred, conf))
    print(f"{file_name} -> Tahmin: {pred} (Guven: %{conf:.1f})")

if images_to_show:
    plt.figure(figsize=(max(len(images_to_show) * 1.5, 4), 2.5))
    for idx, (im, p, c) in enumerate(images_to_show):
        plt.subplot(1, len(images_to_show), idx + 1)
        plt.imshow(im, cmap='gray')
        plt.title(f"{p}\n(%{c:.0f})")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("numara_tahminleri.png")