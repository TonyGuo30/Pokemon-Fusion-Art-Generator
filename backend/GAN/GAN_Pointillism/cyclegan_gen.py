# =============================================================================
# CycleGAN for Style Transfer
# Image Generation using Trained Model
# =============================================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import os
import glob
import sys

# --- 1. Define Model Architecture ---
# same as in GAN_style_transfer.py

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x): 
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)

# --- 2. Configuration & Paths ---

MODEL_DIR = "model"       # Directory containing .pth files (e.g., G_AB_10.pth)
CONTENT_DIR = "content"   # Directory containing input .png images
RESULT_DIR = "result"     # Directory containing generated outputs

# Match the image size used during training
INPUT_SHAPE = (3, 256, 256) 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def run_batch_inference():
    # Get all G_AB models and sort them
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "G_AB_*.pth")))
    extensions = ('.jpg', '.jpeg', '.png')
    content_files = sorted([
        f for f in glob.glob(os.path.join(CONTENT_DIR, "*")) 
        if f.lower().endswith(extensions)])

    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize Generator
    generator = GeneratorResNet(INPUT_SHAPE).to(device)

    # --- Double Loop: Iterate Models -> Iterate Images ---
    # Iterate Models
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".pth", "") 
        print(f"\n---> Loading Model: {model_name}")

        try:
            # Load weights for the current epoch
            generator.load_state_dict(torch.load(model_path, map_location=device))
            generator.eval() # Set to evaluation mode
        except Exception as e:
            print(f"    Failed to load model: {e}. Skipping.")
            continue

        # Create a specific folder for this model epoch inside 'result/'
        current_save_dir = os.path.join(RESULT_DIR, model_name)
        os.makedirs(current_save_dir, exist_ok=True)

        # Iterate Images
        for img_path in content_files:
            img_name = os.path.basename(img_path)
            
            # Load and Preprocess Image
            image = Image.open(img_path)
            if image.mode == 'RGBA':
                bg = Image.new('RGB', image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            else:
                image = image.convert('RGB')

            # Convert to tensor and move to device
            img_tensor = transform(image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output_tensor = generator(img_tensor)

            # Save Result
            save_path = os.path.join(current_save_dir, img_name)
            # normalize=True scales values from (-1, 1) back to (0, 1) for saving
            save_image(output_tensor, save_path, normalize=True)
        
            print(f"    Generated: {save_path}")

    print("\n=== Batch Inference Completed ===")

if __name__ == "__main__":
    # Ensure the main output directory exists
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    run_batch_inference()