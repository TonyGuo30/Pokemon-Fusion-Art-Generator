---------- CycleGAN Style Transfer (Ghibli/Pointillism/Ukiyoe) ----------

This project uses CycleGAN to perform unpaired Image-to-Image translation. It trains a model to translate images from a source domain (e.g., Pokemon Image) to a target domain (e.g., Ukiyoe Style).
You need to create trainA and trainB in the dataset folder first to correctly run cyclegan_train.py.

Directory Structure:
1. dataset/: Training data.
2. trainA: Source domain images.
3. trainB: Target domain images.
4. model/: Stores trained model weights (.pth files).
5. content/: Input images for testing/generation.
6. result/: Output folder for generated stylized images.
7. cyclegan_train.py: Script for training and saving the model.
7. cyclegan_gen.py: Script for generating stylized images.

How to Run:
Phase 1: Training
1. Place your dataset images in dataset/trainA and dataset/trainB. (style A -> B)
2. Start training: python cyclegan_train.py
3. Model weights (checkpoints) will be saved to model/ every 10 epochs (100 epochs in total).

Phase 2: Generation
1. Place the images you want to stylize in content/.
2. Ensure you have at least one trained model (e.g., G_AB_final.pth) in model/. You can generate images of multiple models in one script.
3. Run script: python cyclegan_gen.py
4. Stylized results will be saved in result/ for each model in model/.

Trained Models & Datasets:
We have successfully trained CycleGAN models on three distinct artistic styles using Pokemon Images as the source domain. 
As shown in the model folder, the models have been trained for up to 100 epochs, with checkpoints saved every 10 epochs.
All datasets come from kaggle: https://www.kaggle.com/.

Dataset Configuration:
Source Domain (Content): 
1. Pokemon Images. link to download: https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset
Target Domains (Styles): 
1. Ghibli: Studio Ghibli anime style. link to download: https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images
2. Pointillism: Pointillist art style. link to download: https://www.kaggle.com/datasets/steubk/wikiart?select=Pointillism
3. Ukiyoe: Traditional Japanese woodblock prints. link to download: https://www.kaggle.com/datasets/steubk/wikiart?select=Ukiyo_e

Model Checkpoints
The trained weights are stored in the model/ directory of each corresponding style folder (e.g., GAN_Ukiyoe/model/).
Files are named G_AB_xx.pth (e.g., G_AB_90.pth), representing the generator model at specific training epochs.
Generated results for each epoch are archived in the result/ folder (e.g., result/G_AB_80) for quality comparison.
The author has appended the suffix _best to the result folder that yielded the highest quality generation (e.g., result/G_AB_80_best).