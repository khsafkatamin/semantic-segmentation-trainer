import random
import time
import torch
import torch.cuda.amp as amp
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from models.unet import UNet
from datasets.data_pre_loaders import cityscapes  # or kitti if you want

def load_model(model_path, device):
    model = UNet(input_channels=3, num_classes=30)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, resize_size=(512, 1024)):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def visualize_prediction(image, pred_mask, save_path="predicted_segmentation.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred_mask, cmap='jet', interpolation='nearest')
    plt.axis('off')

    plt.savefig(save_path)
    print(f"Saved prediction visualization as {save_path}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "unet_model.pth"
    model = load_model(model_path, device)

    # Load test dataset (only images list)
    test_dataset = cityscapes.get_dataset(root="datasets/raw_data/cityscapes", split="test", augment=False)
    
    # Pick a random test image
    img_path = random.choice(test_dataset.images)
    print(f"Testing on image: {img_path}")

    # Preprocess image
    input_tensor = preprocess_image(img_path).to(device)

    # Forward pass with mixed precision
    with torch.no_grad():
        start_time = time.time()
        with amp.autocast():
            output = model(input_tensor)  # FP16 where possible
        end_time = time.time()

        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # HxW predicted classes
        fps = 1 / (end_time - start_time)
        print(f"Inference Time: {end_time - start_time:.4f}s, FPS: {fps:.2f}")

    # Load original image for visualization (WxH)
    orig_image = Image.open(img_path).convert("RGB").resize((1024, 512))  # (width, height)

    # Visualize prediction
    visualize_prediction(orig_image, pred)


if __name__ == "__main__":
    main()