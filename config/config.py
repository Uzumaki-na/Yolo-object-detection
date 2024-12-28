import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    NUM_CLASSES = 20
    NUM_BOXES = 2
    SPLIT_SIZE = 7
    IMAGE_SIZE = 224
    
    # Training parameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 135
    NUM_WORKERS = 4
    
    # Dataset paths
    DATASET_PATH = "data/VOC2012"
    TRAIN_DIR = f"{DATASET_PATH}/JPEGImages"
    TRAIN_ANNOTATIONS = f"{DATASET_PATH}/Annotations"
    
    # Transforms
    train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),  # This should always happen first
    A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),  # Change crop size to match IMAGE_SIZE
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_transforms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
