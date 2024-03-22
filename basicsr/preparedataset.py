import os
import shutil

def copy_images(source_dir, target_dir):
    image_extensions = ['.jpg', '.png', '.jpeg']  # Add more extensions if needed

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                os.symlink(source_path, target_path)  # Create a symbolic link

div2k_train_source = "/horizon-bucket/NeuralISP/public_dataset/DIV2K/DIV2K_train_HR/"
div2k_valid_source= "/horizon-bucket/NeuralISP/public_dataset/DIV2K/DIV2K_valid_HR/"
flick2k_source = "/horizon-bucket/NeuralISP/public_dataset/Flickr2k/Flickr2K/Flickr2K_HR/"
df2k_target = "/horizon-bucket/NeuralISP/public_dataset/df2k"

copy_images(div2k_train_source, df2k_target)
copy_images(div2k_valid_source, df2k_target)
copy_images(flick2k_source, df2k_target)
