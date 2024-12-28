import torch
import cv2
import numpy as np
from models.yolo import YOLOv1
from utils.bbox import non_max_suppression, convert_cellboxes
from config.config import Config

class Detector:
    def __init__(self, checkpoint_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.device = Config.DEVICE
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize model
        self.model = YOLOv1(
            split_size=Config.SPLIT_SIZE,
            num_boxes=Config.NUM_BOXES,
            num_classes=Config.NUM_CLASSES
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
                       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
    def preprocess_image(self, image_path):
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        input_image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        # Normalize and convert to tensor
        input_image = input_image / 255.0
        input_image = torch.from_numpy(input_image).float()
        input_image = input_image.permute(2, 0, 1)
        input_image = input_image.unsqueeze(0)
        
        return image, input_image
    
    def detect(self, image_path):
        original_image, input_image = self.preprocess_image(image_path)
        input_image = input_image.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_image)
        
        # Convert cell boxes to bounding boxes
        bboxes = convert_cellboxes(predictions)
        bboxes = non_max_suppression(
            bboxes[0],
            iou_threshold=self.nms_threshold,
            threshold=self.confidence_threshold,
            box_format="midpoint"
        )
        
        return original_image, bboxes
    
    def draw_boxes(self, image, boxes):
        for box in boxes:
            assert len(box) == 6, "box should contain class_pred, confidence, x, y, width, height"
            class_pred = int(box[0])
            conf = box[1]
            x, y, width, height = box[2:]
            
            # Convert to corner format
            x1 = int((x - width/2) * image.shape[1])
            y1 = int((y - height/2) * image.shape[0])
            x2 = int((x + width/2) * image.shape[1])
            y2 = int((y + height/2) * image.shape[0])
            
            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.classes[class_pred]} {conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image