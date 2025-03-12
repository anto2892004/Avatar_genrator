import cv2
import os
from mtcnn import MTCNN

def detect_and_crop_faces(input_folder, output_folder):
    detector = MTCNN()
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)

        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            cropped_face = image[y:y+height, x:x+width]
            output_path = os.path.join(output_folder, f"face_{i}_{img_name}")
            cv2.imwrite(output_path, cropped_face)

    print("Face cropping complete. Check the 'cropped_images' folder.")

detect_and_crop_faces("images/", "cropped_images/")
