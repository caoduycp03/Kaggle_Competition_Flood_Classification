import torch
from model2 import EfficientFlood
from torch.utils.data import DataLoader
from FloodDataset import FloodDataset
from torchvision.transforms import ToTensor,Compose,Resize
from config import ModelConfigs
import multiprocessing
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop
from torchvision.transforms.functional import InterpolationMode
import pandas as pd

if __name__ == '__main__':
    multiprocessing.freeze_support()

    configs = ModelConfigs()
    root = configs.root
    batch_size = configs.batch_size
    train_workers = configs.train_workers
    height = configs.height
    width = configs.width
    checkpoint = configs.checkpoint

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")   

    model = EfficientFlood(num_classes=2).to(device)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model"])
    # print(checkpoint['epoch'])


    # Load test data loader
    resize_size = [480]
    crop_size = [480]
    mean = [0.5, 0.5, 0.5] 
    std = [0.5, 0.5, 0.5]

    transform = Compose([
        Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    test_dataset = FloodDataset(root = root, train=False, transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        num_workers=train_workers,
        drop_last=False,
        shuffle=False
    )

    model.eval()

    pred_df = {'id': [images_path[35:].split('.')[0] for images_path in test_dataset.images_path]}
    pred_df = pd.DataFrame(pred_df)
    all_predictions = []
    for i, images in enumerate(test_dataloader):
        with torch.no_grad():
            images = images.to(device)
            predictions = model(images)
            indices = torch.argmax(predictions.cpu(), dim = 1)
            all_predictions.extend(indices)

    all_predictions = [prediction.item() for prediction in all_predictions]
    pred_df['label'] = all_predictions
    pred_df.to_csv('pred.csv', index='id')
    print('Done')
    
