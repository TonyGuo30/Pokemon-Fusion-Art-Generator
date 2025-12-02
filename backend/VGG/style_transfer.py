# =============================================================================
# Neural Style Transfer Using VGG19
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import matplotlib.pyplot as plt
import os
import glob

# --- 1. Configurations ---

# Device set
device = torch.device("mps")

# Target image size
imsize = 512


# =============================================================================
# --- Tuning Hyperparameters ---
# --------------------------------------------------------------------------
#
STYLE_WEIGHT = 1e6       # Style weight (Try 1e5 to 1e7)
CONTENT_WEIGHT = 1       # Content weight (Usually 1)
TV_WEIGHT = 1e-3         # Total Variation Loss weight (Smooths the image)
NUM_STEPS = 300          # Optimization steps (300-500 is sufficient)

INIT_METHOD = "content"  # "content" (start from content image)

# Layers
CONTENT_LAYERS = ['conv3_2'] 
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
#
# --------------------------------------------------------------------------
# =============================================================================


# --- 2. Image Loading and Preprocessing ---

# VGG19 normalization mean and standard deviation
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def image_loader(image_name, size, background_color=(255, 255, 255)):
    """
    Loads an image, scales it, and converts it.
    """
    image = Image.open(image_name)
    
    # Handle transparent PNGs
    if image.mode == 'RGBA':
        # Create a new solid background image
        new_image = Image.new('RGB', image.size, background_color)
        # Paste the image on top
        new_image.paste(image, mask=image.split()[3])
        image = new_image
    else:
        # Standard conversion for non-RGBA images
        image = image.convert('RGB')
    
    # Scaling logic
    if isinstance(size, int):
        #Resize the shorter side to size, then center crop to (size, size)
        loader = transforms.Compose([
            transforms.Resize(size), 
            transforms.CenterCrop(size),
            transforms.ToTensor()])
    elif isinstance(size, (tuple, list)):
        # Directly resize to (h, w)
        loader = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()])
    else:
        loader = transforms.ToTensor()

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    """
    Converts a Tensor back to a PIL Image for saving
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


# --- 3. Model and Loss Function Definitions ---

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        self.loss_val = self.loss(input, self.target)
        return input

def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    G = torch.mm(features, features.t())
    return G.div(B * C * H * W)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss_val = self.loss(G, self.target)
        return input

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        # Calculate L2 norm of difference between adjacent pixels (Horizontal and Vertical)
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return tv_h + tv_w

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers, style_layers):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    block = 1
    conv_in_block = 1
    
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = f'conv{block}_{conv_in_block}'
            conv_in_block += 1
        elif isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)
            name = f'relu{block}_{conv_in_block-1}'
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{block}'
            block += 1
            conv_in_block = 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn{block}_{conv_in_block-1}'
        else:
            name = f'unknown_{type(layer).__name__}'

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{block}_{conv_in_block-1}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{block}_{conv_in_block-1}", style_loss)
            style_losses.append(style_loss)

    for k in range(len(model) - 1, -1, -1):
        if isinstance(model[k], ContentLoss) or isinstance(model[k], StyleLoss):
            model = model[:(k + 1)]
            break

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# --- 4. Optimization Loop ---
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps, style_weight, content_weight, tv_weight,
                       content_layers, style_layers):
    """
    Performs Neural Style Transfer
    """
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img,
        content_layers, style_layers)

    optimizer = get_input_optimizer(input_img)
    tv_loss_fn = TVLoss().to(device)

    print(f'  Starting optimization... (Total {num_steps} steps)')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            tv_score = 0

            for sl in style_losses:
                style_score += sl.loss_val
            for cl in content_losses:
                content_score += cl.loss_val
            
            # Calculate Total Variation Loss
            tv_score = tv_loss_fn(input_img)

            # Apply weights
            style_score *= style_weight
            content_score *= content_weight
            tv_score *= tv_weight

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"    [Iter {run[0]:>3} / {num_steps}]")
                print(f"      Style Loss: {style_score.item():.2e} "
                      f"Content Loss: {content_score.item():.2e} "
                      f"TV Loss: {tv_score.item():.2e}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


# --- 5. Main Execution ---

def main():
    # =========================================================================
    # --- Folder Paths ---
    #
    CONTENT_FOLDER = "content_images" # content
    STYLE_FOLDER = "style_images"     # style
    OUTPUT_FOLDER = "output_images" # output
    #
    # =========================================================================

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("Loading VGG19 model...")
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    print(f"--- Starting Batch Processing ---")
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 1. Scan all files
    try:
        content_files = [f for f in os.listdir(CONTENT_FOLDER) if f.lower().endswith(supported_extensions) and not f.startswith('.')]
        style_files = [f for f in os.listdir(STYLE_FOLDER) if f.lower().endswith(supported_extensions) and not f.startswith('.')]
        
        if not content_files: print(f"Warning: No content images found in '{CONTENT_FOLDER}'."); return
        if not style_files: print(f"Warning: No style images found in '{STYLE_FOLDER}'."); return

    except Exception as e:
        print(f"Error scanning folders: {e}")
        return
    
    print(f"    Scanning success.")

    # 2. Double loop, iterating through [Style] x [Content] combinations
    
    # Outer loop: Style 
    for style_filename in style_files:
        style_image_path = os.path.join(STYLE_FOLDER, style_filename)
        style_base_name = os.path.splitext(style_filename)[0] 

        # Create a dedicated output folder for this style
        output_folder_name = f"opt_vgg19_{style_base_name}"
        output_dir_path = os.path.join(OUTPUT_FOLDER, output_folder_name)
        os.makedirs(output_dir_path, exist_ok=True)

        print(f"\n--- Processing Combination: [Style: {style_base_name}] ---")
        print(f"    Outputting to: {output_dir_path}")

        # Inner loop: Content
        for content_filename in content_files:
            content_image_path = os.path.join(CONTENT_FOLDER, content_filename)
            print(f"\n  [Content: {content_filename}]")

            try:
                # Step 1: Load Content Image
                content_img = image_loader(content_image_path, imsize)
                
                # Step 2: Load Style Image (Size matches content)
                h, w = content_img.size()[-2:]
                style_img = image_loader(style_image_path, (h, w)) 

            except Exception as e:
                print(f"    Error: Failed to load image: {e}. Skipping {content_filename}.")
                continue
            
            # Step 3: Prepare Input Image
            input_img = content_img.clone()
            
            # Step 4: Run Style Transfer
            output_tensor = run_style_transfer(
                                cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                                num_steps=NUM_STEPS,
                                style_weight=STYLE_WEIGHT,
                                content_weight=CONTENT_WEIGHT,
                                tv_weight=TV_WEIGHT,
                                content_layers=CONTENT_LAYERS,
                                style_layers=STYLE_LAYERS)

            # Step 5: Save Result
            final_image_pil = tensor_to_image(output_tensor)
            
            output_filename = os.path.basename(content_filename)
            base, _ = os.path.splitext(output_filename)
            
            output_filename = f"{base}.png"
            output_save_path = os.path.join(output_dir_path, output_filename)

            final_image_pil.save(output_save_path)
            print(f"  Success! Image saved to: {output_save_path}")
            
        print(f"--- Combination [Style: {style_base_name}] Processing Complete ---")

    print("\n--- All Batch Processing Completed ---")


if __name__ == "__main__":
    main()