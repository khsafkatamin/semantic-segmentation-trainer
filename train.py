import argparse
import torch
from torch.utils.data import ConcatDataset, DataLoader
from datasets.data_pre_loaders import kitti, cityscapes
from loguru import logger

from torchmetrics.classification import MulticlassJaccardIndex

from models.unet import UNet

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Trainer")

    parser.add_argument(
        "--datasets",
        nargs='+',
        choices=["kitti", "cityscapes"],
        required=True,
        help="Datasets to use for training. E.g., --datasets kitti cityscapes"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Base folder where dataset folders are located"
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )


    return parser.parse_args()


def load_datasets(selected_datasets, data_root, split='train', augment=False):
    """
    Load the selected datasets for a specific split (train or val).
    Returns a concatenated dataset if multiple datasets selected.
    """
    dataset_list = []

    for name in selected_datasets:
        if name == "kitti":
            dataset = kitti.get_dataset(root=f"{data_root}/kitti/raw_data", split=split)
        elif name == "cityscapes":
            dataset = cityscapes.get_dataset(root=f"{data_root}/raw_data/cityscapes", split=split, resize_size=(512, 1024), augment=augment)
        else:
            raise ValueError(f"Unsupported dataset: {name}")

        dataset_list.append(dataset)

    if len(dataset_list) == 1:
        return dataset_list[0]
    else:
        return ConcatDataset(dataset_list)


def main():
    args = parse_args()

    logger.info(f"Selected datasets: {args.datasets}")
    train_dataset = load_datasets(args.datasets, args.data_root, split='train', augment=args.augment)
    val_dataset = load_datasets(args.datasets, args.data_root, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    model = UNet(
        input_channels=3,
        num_classes=30,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = 50

    # Initialize mIoU metrics
    train_miou_metric = MulticlassJaccardIndex(num_classes=30, ignore_index=255).to(device)
    val_miou_metric = MulticlassJaccardIndex(num_classes=30, ignore_index=255).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_miou_metric.reset()
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).long()

            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            labels[(labels < 0) | (labels >= 30)] = 255

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Compute train mIoU
            preds = torch.argmax(outputs, dim=1)
            train_miou_metric.update(preds, labels)

        avg_loss = running_loss / len(train_loader)
        train_miou = train_miou_metric.compute().item()
        logger.info(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f} | Train mIoU: {train_miou:.4f}")

        # --- VALIDATION LOOP ---
        model.eval()
        val_running_loss = 0.0
        val_miou_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}"):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device).long()

                if labels.ndim == 4 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                labels[(labels < 0) | (labels >= 30)] = 255

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_miou_metric.update(preds, labels)

        val_avg_loss = val_running_loss / len(val_loader)
        val_miou = val_miou_metric.compute().item()

        logger.info(f"Validation Epoch {epoch + 1} | Loss: {val_avg_loss:.4f} | mIoU: {val_miou:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "unet_model.pth")
    logger.info("Model saved as unet_model.pth")

if __name__ == "__main__":
    main()