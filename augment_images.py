import albumentations as A
import cv2
import os

def augment_images(input_folder, output_folder):
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.GaussNoise(p=0.2)
    ])
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping {img_name}, unable to read file.")
            continue
        augmented = augmentations(image=image)["image"]
        output_path = os.path.join(output_folder, f"aug_{img_name}")
        cv2.imwrite(output_path, augmented)

    print("Data augmentation complete. Check the 'augmented_images' folder.")

# Use your absolute paths
input_folder = r"C:\Users\tejag\Desktop\personalized_avatar\images"
output_folder = r"C:\Users\tejag\Desktop\personalized_avatar\augmented_images"

augment_images(input_folder, output_folder)
