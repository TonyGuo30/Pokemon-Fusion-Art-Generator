# =============================================================================
# CycleGAN for Style Transfer
# Model Training
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import itertools

# --- 1. Configuration & Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 1        # CycleGAN usually works best with batch_size=1
LR = 0.0002           # Learning rate
EPOCHS = 100          # Training epochs
IMG_SIZE = 256        # Standard size for CycleGAN
LAMBDA_CYCLE = 10.0   # Weight for Cycle Consistency Loss
LAMBDA_IDENTITY = 0.5 # Weight for Identity Loss

# Paths
# Structure:
#   dataset/
#       trainA/ (Put all your Source images here)
#       trainB/ (Put all your Style images here)
DATASET_ROOT = "./dataset"

# --- 2. Model Architecture ---

class ResidualBlock(nn.Module):
    """
    Helper block for ResNet Generator
    """
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
    """
    Generator Network
    """
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        # Initial Convolution
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling (Encoder)
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual Blocks (Transformer)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling (Decoder)
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(), # Tanh outputs -1 to 1
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator: Classifies NxN patches of the image as real or fake.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            # Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# --- 3. Dataset ---

class ImageDataset(Dataset):
    """
    Loads images from folder A and folder B.
    """
    def __init__(self, root, transform=None, mode="train"):
        self.transform = transform
        # Create list of files
        self.files_A = sorted([os.path.join(root, f"{mode}A", x) for x in os.listdir(os.path.join(root, f"{mode}A")) if x.endswith(('.png', '.jpg'))])
        self.files_B = sorted([os.path.join(root, f"{mode}B", x) for x in os.listdir(os.path.join(root, f"{mode}B")) if x.endswith(('.png', '.jpg'))])

    def __getitem__(self, index):
        # CycleGAN uses unpaired data, so we use modulus to loop if lists are different lengths
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])

        # Handle Transparenc
        if image_A.mode == 'RGBA':
            background = Image.new('RGB', image_A.size, (255, 255, 255))
            background.paste(image_A, mask=image_A.split()[3])
            image_A = background
        else:
            image_A = image_A.convert("RGB")

        if image_B.mode == 'RGBA':
            background = Image.new('RGB', image_B.size, (255, 255, 255))
            background.paste(image_B, mask=image_B.split()[3])
            image_B = background
        else:
            image_B = image_B.convert("RGB")
            
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ReplayBuffer:
    """
    Stores generated images to stabilize training
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# --- 4. Main Training Loop ---

def train():
    print(f"--- Starting CycleGAN Training on {device} ---")
    
    # 1. Initialize Models
    input_shape = (3, IMG_SIZE, IMG_SIZE)
    
    # G_AB: Pokemon -> Style, G_BA: Style -> Pokemon
    G_AB = GeneratorResNet(input_shape).to(device)
    G_BA = GeneratorResNet(input_shape).to(device)
    
    # D_A: Detect Pokemon, D_B: Detect Style
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    # 2. Losses and Optimizers
    criterion_GAN = nn.MSELoss() 
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    # 3. Data Loading
    transforms_ = [
        transforms.Resize(int(IMG_SIZE * 1.12), Image.BICUBIC),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    # Create dummy folders if they don't exist
    os.makedirs(f"{DATASET_ROOT}/trainA", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/trainB", exist_ok=True)
    
    try:
        dataloader = DataLoader(
            ImageDataset(DATASET_ROOT, transform=transforms.Compose(transforms_)),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        print(f"Data loaded: {len(dataloader)} batches.")
    except Exception as e:
        print(f"Error loading data: {e}.")
        return

    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 4. Training
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device) # Pokemon
            real_B = batch["B"].to(device) # Style

            # valid (1) and fake (0) labels
            valid = torch.ones(real_A.size(0), *D_A(real_A).size()[1:], requires_grad=False).to(device)
            fake = torch.zeros(real_A.size(0), *D_A(real_A).size()[1:], requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            G_AB.train(); G_BA.train()
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2 * LAMBDA_IDENTITY

            # GAN loss
            fake_B = G_AB(real_A) # Generated Style
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            
            fake_A = G_BA(real_B) # Generated Pokemon
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B) # Pokemon -> Style -> Pokemon
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            
            recov_B = G_AB(fake_A) # Style -> Pokemon -> Style
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2 * LAMBDA_CYCLE

            # Total Generator Loss
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator A
            # ---------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------------------
            #  Train Discriminator B
            # ---------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Logging
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] "
                      f"[G loss: {loss_G.item():.4f}]")

        # Save Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(G_AB.state_dict(), f"model/G_AB_{epoch}.pth")
            print(f"Checkpoint saved: model/G_AB_{epoch}.pth")

    # Save the final result
    torch.save(G_AB.state_dict(), f"model/G_AB_final.pth")
    print(f"Final model saved: model/G_AB_final.pth")

if __name__ == "__main__":
    train()