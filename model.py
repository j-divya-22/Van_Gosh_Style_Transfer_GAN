
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# --------------------------------------------------
# CycleGAN residual block
# --------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=0, bias=True
            ),
            nn.BatchNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=0, bias=True
            ),
            nn.BatchNorm2d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# --------------------------------------------------
# Generator architecture
# --------------------------------------------------

class CompatibleResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=True),
            nn.BatchNorm2d(ngf, affine=False),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for i in range(2):
            mult = 2 ** i
            layers.extend([
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True
                ),
                nn.BatchNorm2d(ngf * mult * 2, affine=False),
                nn.ReLU(inplace=True),
            ])

        # Nine residual blocks
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf * 4))

        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            layers.extend([
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True
                ),
                nn.BatchNorm2d(ngf * mult // 2, affine=False),
                nn.ReLU(inplace=True),
            ])

        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Load generator weights
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FINE_TUNED_MODEL_PATH = (
    BASE_DIR / "models" / "fine_tuned" / "vangogh_generator.pth"
)


def load_generator(model_path: Path | None = None):
    model_path = model_path or FINE_TUNED_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CompatibleResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # Strict loading detects architecture mismatches.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# Convert uploaded photo
# --------------------------------------------------

def convert_image(image: Image.Image, model, device):
    original_size = image.size
    scale = 256 / max(original_size)
    target_width = max(4, round(original_size[0] * scale / 4) * 4)
    target_height = max(4, round(original_size[1] * scale / 4) * 4)

    preprocess = transforms.Compose([
        transforms.Resize(
            (target_height, target_width),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
    ])

    input_tensor = preprocess(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    output = (
        output.squeeze(0).cpu().clamp(-1, 1) + 1
    ) / 2

    result = transforms.ToPILImage()(output)
    return result.resize(original_size, Image.Resampling.LANCZOS)
