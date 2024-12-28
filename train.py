import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from config.config import Config
from models.yolo import YOLOv1
from models.loss import YOLOLoss
from data.dataset import VOCDataset
import matplotlib.pyplot as plt
from utils.visualization import plot_image
import warnings
import os
warnings.filterwarnings("ignore")

def train_fn(train_loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)
            
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
        
    return sum(mean_loss)/len(mean_loss)

def main():

    os.makedirs("checkpoints", exist_ok=True)

    # Initialize model and move to GPU
    model = YOLOv1(
        split_size=Config.SPLIT_SIZE,
        num_boxes=Config.NUM_BOXES,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Load Dataset
    train_dataset = VOCDataset(
        Config.TRAIN_DIR,
        Config.TRAIN_ANNOTATIONS,
        transform=Config.train_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, Config.DEVICE)
        
        print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Average Loss: {avg_loss:.5f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoints/checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()