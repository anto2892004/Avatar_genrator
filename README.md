

# Avatar Generator

A Python-based tool for generating personalized, stylized avatars using Stable Diffusion with LoRA (Low-Rank Adaptation) fine-tuning. The project supports data augmentation, face detection, and embedding similarity comparison to create and evaluate unique avatars for profile pictures or creative projects.

## 🚀 Features

- Generate avatars using Stable Diffusion with fine-tuned LoRA weights
- Data augmentation with transformations (e.g., flips, brightness adjustments, noise)
- Face detection and cropping using MTCNN
- Embedding similarity comparison for identity verification
- Export avatars as PNG files
- GPU acceleration support with CUDA

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Required libraries:
  - `torch`
  - `diffusers`
  - `peft`
  - `transformers`
  - `accelerate`
  - `huggingface_hub`
  - `albumentations`
  - `opencv-python` (`cv2`)
  - `mtcnn`
  - `scikit-learn`
  - `numpy`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anto2892004/Avatar_genrator.git
   cd Avatar_genrator
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv avatar_env
   source avatar_env/bin/activate  # On Windows: avatar_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch diffusers peft transformers accelerate huggingface_hub albumentations opencv-python mtcnn scikit-learn numpy
   ```

4. **Verify CUDA availability** (if using GPU):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Ensure this returns `True` if you have a CUDA-compatible GPU.

## 🛠 Usage

### 1. Data Augmentation
Augment input images to create varied training data:
- Place input images in the `images/` folder.
- Run the augmentation script to generate transformed images, saved to `augmented_images/`:
  ```bash
  python augment_images.py
  ```

### 2. Face Detection and Cropping
Detect and crop faces from input images:
- Ensure input images are in the `images/` folder.
- Run the face detection script to save cropped faces to `cropped_images/`:
  ```bash
  python detect_faces.py
  ```

### 3. Fine-Tune Stable Diffusion with LoRA
Train the LoRA adapter for the Stable Diffusion U-Net:
- Run the training script to fine-tune and save LoRA weights to `lora_unet/`:
  ```bash
  python train_lora.py
  ```
- **Note**: The provided training script uses a dummy loop. Replace it with your dataset and training logic for actual fine-tuning.

### 4. Check LoRA Weights
Verify the presence of LoRA weights and retrain if necessary:
- Run the script to check for weights in `lora_unet/`:
  ```bash
  python check_lora.py
  ```
- If weights are missing, the script automatically installs dependencies and runs `train_lora.py`.

### 5. Generate Avatars
Generate a personalized avatar using the fine-tuned model:
- Run the generation script to create an avatar, saved as `generated_avatar.png`:
  ```bash
  python generate_avatar.py
  ```
- Customize the prompt in `generate_avatar.py` (e.g., "A futuristic portrait of a young person, highly detailed, digital painting").

### 6. Evaluate Similarity
Compare embeddings of generated avatars with original images:
- Run the similarity script to compute cosine similarity between embeddings:
  ```bash
  python similarity_check.py
  ```
- **Note**: Update the script with actual image embeddings for meaningful results.

## 📂 Project Structure

```
Avatar_genrator/
├── images/                 # Input images for augmentation and face detection
├── augmented_images/       # Augmented images output
├── cropped_images/         # Cropped face images output
├── lora_unet/             # LoRA weights for fine-tuned U-Net
├── augment_images.py       # Script for data augmentation
├── check_lora.py          # Script to verify or retrain LoRA weights
├── detect_faces.py         # Script for face detection and cropping
├── train_lora.py          # Script for LoRA fine-tuning
├── generate_avatar.py      # Script for avatar generation
├── similarity_check.py     # Script for embedding similarity comparison
├── generated_avatar.png    # Generated avatar output
└── README.md              # Project documentation


