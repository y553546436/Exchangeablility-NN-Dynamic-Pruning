import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import math
import os
from tqdm import tqdm
from dataclasses import dataclass
import data
import models.models as models
from models.models import save_model, save_model_params, load_model
import copy
import wandb
import random
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
from config import DEVICE

criterion = nn.CrossEntropyLoss()

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del outputs, loss
    return loss_total / total, correct / total

def evaluate_top1_top5_accuracy(model, dataloader):
     """Evaluate the model and compute top-1 and top-5 accuracy"""
     model.eval()
     correct_top1 = 0
     correct_top5 = 0
     total = 0
     
     with torch.no_grad():
         for data in tqdm(dataloader):
             images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
             outputs = model(images)
             
             # Top-1 accuracy
             _, predicted = outputs.max(1)
             correct_top1 += predicted.eq(labels).sum().item()
             
             # Top-5 accuracy
             _, top5_predicted = outputs.topk(5, 1, largest=True, sorted=True)
             correct_top5 += top5_predicted.eq(labels.view(-1, 1).expand_as(top5_predicted)).sum().item()
             
             total += labels.size(0)
     
     top1_accuracy = 100.0 * correct_top1 / total
     top5_accuracy = 100.0 * correct_top5 / total
     
     return top1_accuracy, top5_accuracy

def get_model_outputs_labels(model, dataloader):
    output_list = []
    label_list = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            output_list.append(outputs)
            label_list.append(labels)
    return torch.cat(output_list), torch.cat(label_list)

def train_model(model, train_loader, val_loader, test_loader, num_epochs, optimizer="AdamW", lr=5e-3, momentum=0.9, weight_decay=1e-4, scheduler="OneCycleLR"):
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "StepLR":
        sclr = StepLR(optimizer, step_size=20, gamma=0.1)
    elif scheduler == "ReduceLROnPlateau":
        sclr = ReduceLROnPlateau(optimizer, mode='max',
                                factor=0.1,
                                patience=5,
                                )
    elif scheduler == "OneCycleLR":
        sclr = OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch = len(train_loader),
                                pct_start=0.1, anneal_strategy='cos')

    best_val_acc = 0.0
    best_model_params = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_total, train_correct = 0, 0
        for i, data in tqdm(enumerate(train_loader)):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler == "OneCycleLR":
                sclr.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            del loss, outputs

        epoch_loss = running_loss / (i + 1)
        train_acc = 100. * train_correct / train_total
        val_loss, val_acc = evaluate_accuracy(model, val_loader)
        test_loss, test_acc = evaluate_accuracy(model, test_loader)
        print("[{}] train accuracy = {}, validation accuracy = {}, test accuracy = {}, train loss = {}, validation loss = {}, test loss = {}".format(epoch + 1, train_acc, val_acc, test_acc, epoch_loss, val_loss, test_loss))
        wandb.log({"train_accuracy": train_acc, "validation_accuracy": val_acc, "test_accuracy": test_acc, "train loss": epoch_loss, "validation loss": val_loss, "test loss": test_loss, "lr": sclr.get_last_lr()[0]})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = copy.deepcopy(model.state_dict())

        if scheduler == "ReduceLROnPlateau":
            sclr.step(metrics=val_acc)
        elif scheduler == "StepLR":
            sclr.step()
    
    return best_model_params


def iterative_train_prune_save(model_name, model, train_loader, val_loader, test_loader, num_epochs=30, optimizer="AdamW", lr=0.005, momentum=0.9, weight_decay=0.0001, scheduler="OneCycleLR", finished_iters=0):
    # prune_rates = np.linspace(0.0, target_prune_rate, tune_iters)
    remain_rate = 1
    iters = 80
    decay_rate = 0.95
    if finished_iters > 0:
        for i in range(finished_iters):
            if i == finished_iters - 1:
                prune_rate = 1 - remain_rate
                new_model_name = f"{model_name}-0.95decay-prune-iter-{i}"
                model = load_model(new_model_name, model, prune_rate)
            remain_rate *= decay_rate

    from static_prune.l1_pruner import L1Pruner
    pruner = L1Pruner()
    for i in range(finished_iters, iters):
        prune_rate = 1 - remain_rate
        pruner.prune(model, prune_rate)

        wandb.init(
            # set the wandb project where this run will be logged
            project=f"{model_name}-fine-tuning-pruning-0.95decay",
            name=f"{model_name}-prune-iter-{i}",
            # track hyperparameters and run metadata
            config={
                "prune_rate": prune_rate,
                "prune_iter": i,
                "learning_rate": lr,
                "optimizer": optimizer,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "epochs": num_epochs,
            }
        )
        best_params = train_model(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, optimizer=optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay, scheduler=scheduler)
        model.load_state_dict(best_params)
        # Finish the current wandb run
        wandb.finish()
        remain_rate *= decay_rate
        new_model_name = f"{model_name}-0.95decay-prune-iter-{i}"
        save_model(new_model_name, model, prune_rate)

def train_and_save(model_name, model, train_loader, val_loader, test_loader, num_epochs, optimizer="AdamW", lr=0.005, momentum=0.9, weight_decay=0.0001, scheduler=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{model_name}",
        name=f"{model_name}",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "epochs": num_epochs,
            "scheduler": scheduler,
        }
    )
    best_params = train_model(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, optimizer=optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay, scheduler=scheduler)
    save_model_params(model_name, best_params)
    wandb.finish()


def sweep_models(train_loader, val_loader, test_loader):
    num_epochs = 100
    optimizer = "SGD"
    momentums = [0.7, 0.8, 0.9]
    weight_decay = 2e-4
    lrs = [0.001, 0.005, 0.01, 0.02]
    c2s = [10 * i for i in range(10, 31, 5)]
    c3s = [10 * i for i in range(10, 31, 5)]
    total = 1000000 # 27 * c1 + 9 * c1 * c2 + 9 * c2 * c3 + 10 * c3 = total
    p1s = [0.1, 0.2, 0.3]
    p2s = [0.25, 0.3, 0.35]
    p3s = [0.3, 0.35, 0.4]
    import itertools
    # Create the product and shuffle it randomly
    all_combinations = list(itertools.product(momentums, lrs, c2s, c3s, p1s, p2s, p3s))
    random.shuffle(all_combinations)
    for momentum, lr, c2, c3, p1, p2, p3 in all_combinations:
        c1 = int((total - 10 * c2 - 9 * c2 * c3) / (27 + 9 * c2))
        if c1 > 500:
            continue
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="cifar10-3-conv-layer-sweep-1m-data-aug-global-pooling",

            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "momentum": momentum,
            "weight_decay": weight_decay,   
            "optimizer": optimizer,
            "dataset": "CIFAR10",
            "epochs": num_epochs,
            }
        )
        model = models.build_parameterized_cifar10_3_layer_conv_mlp(c1, c2, c3, p1, p2, p3).to(DEVICE).train()
        model_id = ModelID(0, f"cifar10_conv_data_aug_c1_{c1}_p1_{p1}_c2_{c2}_p2_{p2}_c3_{c3}_p3_{p3}_{momentum}_{lr}")
        train_and_save(model, model_id, train_loader, val_loader, test_loader, num_epochs=num_epochs, optimizer=optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay)
        wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(42)

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use (AdamW, Adam, SGD)')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--momentum', type=float, default=0.8, help='Momentum (for SGD)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default="OneCycleLR", help='lr scheduler')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader = data.get_cifar10_data(batch_size=args.bs)
    model_name, model = models.build_vgg11_bn_cifar10()
    model = model.to(DEVICE).train()
    train_and_save(model_name, model, train_loader, val_loader, test_loader, num_epochs=args.epochs, optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, scheduler=args.scheduler)
    iterative_train_prune_save(model_name, model, train_loader, val_loader, test_loader, num_epochs=args.epochs, optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, scheduler=args.scheduler)

    model_name, model = models.build_resnet18_cifar10()
    model = model.to(DEVICE).train()
    train_and_save(model_name, model, train_loader, val_loader, test_loader, num_epochs=args.epochs, optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, scheduler=args.scheduler)