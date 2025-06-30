import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.optim as optim
import models.models as models
from models.models import save_model_params
from torch_geometric.data.storage import GlobalStorage
import torch.serialization
from torch.optim.lr_scheduler import OneCycleLR
from gnn_utils import get_gnn_dataset, gnn_eval

from tqdm import tqdm
import argparse
import numpy as np

from config import DEVICE

import wandb

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train_step(model, loader, optimizer, scheduler, task_type):
    model.train()

    total_loss, total_count = 0, 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(DEVICE)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch.num_graphs
            total_count += batch.num_graphs

    return total_loss / total_count if total_count > 0 else 0.0

def train(epochs, model, train_loader, val_loader, test_loader, optimizer, scheduler, task_type, dataset, evaluator):
    best_params = None
    best_val_metric = -float('inf') if 'classification' in task_type else float('inf')

    for epoch in range(epochs):
        print(f"=== Epoch {epoch} ===")
        avg_loss = train_step(model, train_loader, optimizer, scheduler, task_type)

        train_res = gnn_eval(model, train_loader, evaluator)
        val_res = gnn_eval(model, val_loader, evaluator)
        test_res = gnn_eval(model, test_loader, evaluator)

        print({'Train': train_res, 'Validation': val_res, 'Test': test_res})

        if 'classification' in task_type:
            if val_res[dataset.eval_metric] > best_val_metric:
                best_val_metric = val_res[dataset.eval_metric]
                best_params = model.state_dict()
        else:
            if val_res[dataset.eval_metric] < best_val_metric:
                best_val_metric = val_res[dataset.eval_metric]
                best_params = model.state_dict()

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_metric": train_res[dataset.eval_metric],
            "val_metric": val_res[dataset.eval_metric],
            "test_metric": test_res[dataset.eval_metric],
        })
    
    return best_params

if __name__ == "__main__":
    # Override DEVICE to use CPU
    DEVICE = torch.device('cpu')

    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs for training')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name')
    args = parser.parse_args()

    wandb.login()

    wandb.init(
        project="GNN",
        name=f"{args.dataset}-GCN",
        config={
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "num_layer": args.num_layer,
            "emb_dim": args.emb_dim,
            "epochs": args.epochs,
            "dataset": args.dataset,
        }
    )

    # Use the safe loading function instead of direct instantiation
    dataset, train_loader, val_loader, test_loader, evaluator = get_gnn_dataset(args.dataset, args.bs)
    # model = GNN(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = (args.vn.lower() == 'true')).to(device)
    model_name, model = models.build_gnn(dataset.num_tasks, args.num_layer, args.emb_dim)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch = len(train_loader),
                                pct_start=0.1, anneal_strategy='cos')

    best_params = train(args.epochs, model, train_loader, val_loader, test_loader, optimizer, scheduler, dataset.task_type, dataset, evaluator)

    save_model_params(model_name, best_params)

    wandb.finish()
