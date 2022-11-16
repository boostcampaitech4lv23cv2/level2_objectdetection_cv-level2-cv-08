from tqdm import tqdm
import os
import argparse
from datetime import datetime

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from Modules import *

def train_fn(num_epochs, train_data_loader, optimizer, model, device):
    check_point_hash = datetime.now().microsecond % 1000
    best_loss = 1000
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = f'./checkpoints/faster_rcnn_torchvision_checkpoints_{check_point_hash}.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=int, default=0.005)
    parser.add_argument('--weight_decay', type=int, default=0.0005)
    parser.add_argument('--annotation', type=str, default='../../dataset/train.json')
    parser.add_argument('--data_dir', type=str, default='../../dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=11)

    args = parser.parse_args()
    
    annotation = args.annotation # annotation 경로
    data_dir = args.dataset # data_dir 경로
    train_dataset = CustomDataset(annotation, data_dir, get_train_transform()) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    device = args.device
    print(device)
    
    # torchvision model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11 # class 개수= 10 + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    num_epochs = args.epochs
    
    # training
    train_fn(num_epochs, train_data_loader, optimizer, model, device)