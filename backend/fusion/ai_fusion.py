# ai_fusion.py
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch, argparse, os

def main():
    parser = argparse.ArgumentParser(description="AI Pokémon Fusion via Stable Diffusion")
    parser.add_argument("--a", required=True, help="Path to Pokémon A image (base image)")
    parser.add_argument("--b", required=True, help="Path to Pokémon B image (style or secondary influence)")
    parser.add_argument("--out", default="fusion_ai.png", help="Output filename")
    parser.add_argument("--prompt", default="fusion creature combining both Pokémon in anime style",
                        help="Text prompt guiding the fusion")
    parser.add_argument("--strength", type=float, default=0.75,
                        help="How strongly to modify the base image (0.3–0.9 typical)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="How much to follow the text prompt (7–12 typical)")
    args = parser.parse_args()

    # ---- Load images ----
    a_img = Image.open(args.a).convert("RGB")
    b_img = Image.open(args.b).convert("RGB")

    # Resize both to same size for consistency
    b_img = b_img.resize(a_img.size)

    # ---- Load pre-trained diffusion model ----
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Prepare fusion prompt ----
    prompt = f"a fusion creature combining {os.path.splitext(os.path.basename(args.a))[0]} and {os.path.splitext(os.path.basename(args.b))[0]}, {args.prompt}"

    # ---- Run image-to-image generation ----
    print(f"Generating fusion with strength={args.strength}, guidance={args.guidance}")
    result = pipe(
        prompt=prompt,
        image=a_img,
        strength=args.strength,
        guidance_scale=args.guidance
    ).images[0]

    # ---- Save result ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.save(args.out)
    print(f"✅ Fusion saved to {args.out}")

if __name__ == "__main__":
    main()
