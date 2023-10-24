import os
import torch
import torch.nn as nn
from FloodDataset import FloodDataset
from model2 import EfficientFlood
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop, ColorJitter, RandomHorizontalFlip
from torch.utils.data import DataLoader 
from torch.utils.data import random_split 
import numpy as np 
from tqdm.autonotebook import tqdm 
from torch.utils.tensorboard import SummaryWriter 
import shutil 
import warnings 
warnings.simplefilter("ignore") 
from config import ModelConfigs 
import numpy as np 
from Early_Stopping import EarlyStopping 
import torch.nn.functional as F 
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import classification_report, f1_score, accuracy_score, average_precision_score, confusion_matrix
from torchvision.ops import sigmoid_focal_loss
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    configs = ModelConfigs()
    root = configs.root
    num_epochs = configs.epochs
    batch_size = configs.batch_size
    train_workers = configs.train_workers
    height = configs.height
    width = configs.width
    learning_rate = configs.learning_rate
    logging = configs.logging
    trained_models = configs.trained_models
    checkpoint = configs.checkpoint
        
    augment_transform = Compose(
            # [ColorJitter(brightness=0.2, 
            #             contrast=0.5,
            #             saturation=0.3), 
            [RandomHorizontalFlip()]
            )
    
    resize_size = [480]
    crop_size = [480]
    mean = [0.5, 0.5, 0.5] 
    std = [0.5, 0.5, 0.5]

    transform = Compose([
            Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean, std=std)])

    #split train/val dataset
    dataset = FloodDataset(root = root, train=True, transform=transform)
    
    # Set the random seed
    torch.manual_seed(43)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

    if os.path.isdir(logging):
        shutil.rmtree(logging)
    if not os.path.isdir(trained_models):
        os.mkdir(trained_models)
    writer = SummaryWriter(logging)


    model = EfficientFlood().to(device)
    for name, param in model.named_parameters():
        if 'fc1d' not in name and 'fc2d' not in name and 'fc3d' not in name and "features.8." not in name and "features.7." not in name:
            param.requires_grad = False

    criterion = sigmoid_focal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]) 
    else:
        start_epoch = 0
        best_acc = 0
        best_map = 0
    num_iters = len(train_dataloader)

    #set early stopping
    early_stopping = EarlyStopping(patience=80, path='{}/best_loss.pt'.format(trained_models))

    #start training and validating
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        sum_loss = 0
        for iter, (images, labels) in enumerate(progress_bar):
            images = augment_transform(images)
            labels = F.one_hot(labels, num_classes=2).float()
            images = images.to(device)
            labels = labels.to(device)
            #forward
            optimizer.zero_grad()
            outputs = model(images)
            loss_value = criterion(outputs, labels, alpha=0.67, gamma=2, reduction='mean')
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss{:3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            sum_loss += loss_value
            #backward
            loss_value.backward()  
            optimizer.step()
        print(' Avg training loss', sum_loss/(len(train_dataset)//batch_size))

        sum_loss  = 0
        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(val_dataloader):
            all_labels.extend(labels)
            labels = F.one_hot(labels, num_classes=2).float()            
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predictions = model(images)
                loss_value = criterion(predictions, labels, alpha=0.67, gamma=2, reduction='mean')
                sum_loss += loss_value
                indices = torch.argmax(predictions.cpu(), dim = 1)
                all_predictions.extend(indices)
        loss_value = sum_loss/(len(val_dataset)//batch_size)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        f1 = f1_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, all_predictions)
        map = average_precision_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch + 1)
        print("Epoch{}: Accuracy: {}. F1: {}. MAP: {}".format(epoch+1, acc, f1, map))
        print(classification_report(all_labels, all_predictions))
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=[0,1], epoch=epoch+1)

        checkpoint = {
            "epoch": epoch + 1,
            "best_acc" : best_acc,
            "best_map" : best_map,            
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(trained_models))

        if acc >= best_acc:
            best_acc = acc
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc" : best_acc,
                "best_map" : best_map,            
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_acc.pt".format(trained_models)) 

        if f1 >= best_map:
            best_map = f1
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc" : best_acc,
                "best_map" : best_map,            
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_map.pt".format(trained_models))

        print(' AVG-VAL Loss Value', loss_value)
        early_stopping(loss_value, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break