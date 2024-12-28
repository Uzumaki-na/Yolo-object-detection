import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Reshape predictions to [batch, S*S*(B*5+C)]
        predictions = predictions.reshape(batch_size, self.split_size * self.split_size, -1)
        targets = targets.reshape(batch_size, self.split_size * self.split_size, -1)
        
        # Calculate IoU for both predicted boxes
        iou_b1 = self.intersection_over_union(predictions[..., 1:5], targets[..., 1:5])
        iou_b2 = self.intersection_over_union(predictions[..., 6:10], targets[..., 1:5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Find best box
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = targets[..., 0].unsqueeze(-1)  # Identity of object i
        
        # ======================== #
        #   FOR BOX COORDINATES   #
        # ======================== #
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 6:10]
                + (1 - best_box) * predictions[..., 1:5]
            )
        )
        
        box_targets = exists_box * targets[..., 1:5]
        
        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N,S,S,4) -> (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        
        # ==================== #
        #   FOR OBJECT LOSS   #
        # ==================== #
        pred_box = (
            best_box * predictions[..., 5:6] + (1 - best_box) * predictions[..., 0:1]
        )
        
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 0:1])
        )
        
        # ======================= #
        #   FOR NO OBJECT LOSS   #
        # ======================= #
        # (N,S*S,1) -> (N,S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 0:1], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 0:1], start_dim=1)
        )
        
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 5:6], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 0:1], start_dim=1)
        )
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # (N,S,S,20) -> (N*S*S,20)
        class_loss = self.mse(
                    # Change from predictions[..., 10:] to get the right slice
        torch.flatten(exists_box * predictions[..., (5 * self.num_boxes):], end_dim=-2),
        # Change from targets[..., 5:] to get the right slice
        torch.flatten(exists_box * targets[..., (5 * self.num_boxes):], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss
    
    def intersection_over_union(self, boxes_preds, boxes_labels):
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        
        return intersection / (box1_area + box2_area - intersection + 1e-6)