import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config.config import Config  

class VOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
                       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        xml_path = os.path.join(self.annotations_dir, 
                               self.image_files[idx].replace('.jpg', '.xml'))
        
        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Parse annotations
        boxes, class_labels = self._parse_annotation(xml_path)
        
        # Apply transforms
        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes, 
                                         class_labels=class_labels)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]
            class_labels = augmentations["class_labels"]
        
        # Convert to tensor format
        target = self._create_target(boxes, class_labels)
        
        return image, target
    
    def _parse_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        class_labels = []
        
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        
        for obj in root.findall('object'):
            label = self.classes.index(obj.find('name').text)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            boxes.append([x_center, y_center, w, h])
            class_labels.append(label)
            
        return boxes, class_labels
    
    def _create_target(self, boxes, class_labels):
        target = torch.zeros((Config.SPLIT_SIZE, Config.SPLIT_SIZE, 
                            5 * Config.NUM_BOXES + Config.NUM_CLASSES))
        
        for box, class_label in zip(boxes, class_labels):
            x, y, width, height = box
            i, j = int(Config.SPLIT_SIZE * y), int(Config.SPLIT_SIZE * x)
            x_cell, y_cell = Config.SPLIT_SIZE * x - j, Config.SPLIT_SIZE * y - i
            
            if target[i, j, 0] == 0:  # If no object
                target[i, j, 0] = 1
                target[i, j, 1:5] = torch.tensor([x_cell, y_cell, width, height])
                target[i, j, 5 + class_label] = 1
                
        return target