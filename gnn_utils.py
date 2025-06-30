import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.optim as optim
import models.models as models
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization
from torch.optim.lr_scheduler import OneCycleLR
# Add required classes to safe globals for PyTorch 2.7 compatibility
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

from tqdm import tqdm
import argparse
import numpy as np

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def gnn_eval(model, loader, evaluator):
    DEVICE = torch.device('cpu')
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(DEVICE)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def gnn_my_eval(model, loader, evaluator, confidences):
    DEVICE = torch.device('cpu')
    model.eval()
    model.my_eval(confidences)
    y_true = []
    y_pred = []
    total = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(DEVICE)
        total += batch.y.shape[0]

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    flops = model.gather_flops()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), flops / total

def safe_load_dataset(name):
    """Safely load OGB dataset with weights_only=False for PyTorch 2.7 compatibility"""
    dataset = PygGraphPropPredDataset(name=name)
    # Override the load method to use weights_only=False
    dataset.data, dataset.slices = torch.load(dataset.processed_paths[0], weights_only=False)
    return dataset

def get_gnn_dataset(name="ogbg-molhiv", bs=64):
    dataset = safe_load_dataset(name)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=bs, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=bs, shuffle=False)
    evaluator = Evaluator(name)
    return dataset, train_loader, val_loader, test_loader, evaluator
