---------- VGG19 Style Transfer ----------

This project implements Neural Style Transfer using a pre-trained VGG19 network. It optimizes a content image to match the artistic style of a reference image.

Directory Structure:
1. content_images/: Place your input photos here (e.g., .jpg, .png).
2. style_images/: Place your style reference artworks here.
3. output_images/: Generated results will be saved here.
4. style_transfer.py: The main script for batch processing.

How to Run:
1. Prepare Images: Put your content images in content_images/ and style images in style_images/.
2. Run Script: python style_transfer.py
3. Check Results: The script will iterate through every combination of content and style. Find the results in output_images/.

Note: You can adjust parameters (weights, image size) directly inside style_transfer.py.