import torch
import torch.nn as nn
import torchvision.models as models

class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        
        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b1(pretrained=True)
        
        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # YOLO head
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, (5 * num_boxes + num_classes), kernel_size=1),
        )
        
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        
        # Reshape to match YOLO output format
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, self.split_size, self.split_size, 
                  (5 * self.num_boxes + self.num_classes))
        
        return x