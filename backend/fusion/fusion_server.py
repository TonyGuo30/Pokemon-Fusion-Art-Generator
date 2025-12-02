from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import os
import tempfile
from diffusers import StableDiffusionImg2ImgPipeline

app = Flask(__name__)
CORS(app)

# Configure gallery folder
GALLERY_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'gallery')

print("Gallery Folder", GALLERY_FOLDER)

# Load model once at startup
print("Loading Stable Diffusion model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded!")

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Return list of sample images"""
    try:
        files = [f for f in os.listdir(GALLERY_FOLDER) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return jsonify({'files': sorted(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples/<filename>', methods=['GET'])
def get_sample_image(filename):
    """Serve a sample image"""
    print(f"Serving sample image: {filename}")
    print(f"GALLERY_FOLDER: {GALLERY_FOLDER}")

    try:
        return send_from_directory(GALLERY_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/fuse', methods=['POST'])
def fuse_images():
    try:
        # Check if using sample images or uploads
        use_samples = request.form.get('use_samples', 'false') == 'true'
        
        if use_samples:
            # Load from samples folder
            filename_a = request.form.get('filenameA')
            filename_b = request.form.get('filenameB')
            
            if not filename_a or not filename_b:
                return jsonify({'error': 'Both filenames required'}), 400
            
            a_img = Image.open(os.path.join(GALLERY_FOLDER, filename_a)).convert("RGB")
            b_img = Image.open(os.path.join(GALLERY_FOLDER, filename_b)).convert("RGB")
        else:
            # Load from uploaded files
            image_a = request.files.get('imageA')
            image_b = request.files.get('imageB')
            
            if not image_a or not image_b:
                return jsonify({'error': 'Both images required'}), 400
            
            a_img = Image.open(image_a.stream).convert("RGB")
            b_img = Image.open(image_b.stream).convert("RGB")
        
        prompt = request.form.get('prompt', 'fusion creature combining both Pokémon in anime style')
        strength = float(request.form.get('strength', 0.75))
        guidance = float(request.form.get('guidance', 7.5))

        # Resize images
        b_img = b_img.resize(a_img.size)

        # Generate fusion
        result = pipe(
            prompt=prompt,
            image=a_img,
            strength=strength,
            guidance_scale=guidance
        ).images[0]

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            result.save(tmp.name)
            tmp_path = tmp.name

        return send_file(tmp_path, mimetype='image/png')

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)