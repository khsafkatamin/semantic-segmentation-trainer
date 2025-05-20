# Semantic Segmentation Trainer

This repository is used to train semantic segmentation models (e.g., U-Net) using PyTorch in a GPU-enabled Docker environment.

## 🔧 Build the Docker Image

```bash
docker build -t semantic-seg-trainer .
```

## 🚀 Run the Docker Container

```bash
docker run --rm -it --gpus all -v "$(pwd)":/workspace -w /workspace semantic-seg-trainer
```