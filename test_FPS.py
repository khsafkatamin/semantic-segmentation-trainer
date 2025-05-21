import time
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from models.unet import UNet
from datasets.data_pre_loaders import cityscapes  # or kitti

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

def main(use_amp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "unet_model.pth"
    model = load_model(model_path, device)

    test_dataset = cityscapes.get_dataset(root="datasets/raw_data/cityscapes", split="test", augment=False)
    image_paths = test_dataset.images
    total_images = len(image_paths)

    total_time = 0.0
    print(f"Running inference on {total_images} images {'with AMP' if use_amp else 'without AMP'}...\n")

    for img_path in tqdm(image_paths, desc="Processing", unit="img"):
        input_tensor = preprocess_image(img_path).to(device)

        start_time = time.time()
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(input_tensor)
            else:
                output = model(input_tensor)
            _ = torch.argmax(output, dim=1)
        total_time += time.time() - start_time

    fps = total_images / total_time
    print(f"\nProcessed {total_images} images in {total_time:.2f} seconds.")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    # Change to True to test AMP
    main(use_amp=True)