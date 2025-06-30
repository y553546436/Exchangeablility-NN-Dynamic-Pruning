import torch
from config import DEVICE
from utils.confidences import StatsTestConfidence, ThresholdConfidence
import models.models as models
from models.models import load_model, load_iterative_pruned_model, get_iterative_pruned_model_name
import data
import optuna
import pickle
import os
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

stats_color = '#e74c3c'
thr_color = '#f39c12'
baseline_color = '#2ecc71'

def load_trial_csv(study_name):
    import csv
    with open(f"optuna_results/{study_name}_trials.csv", "r") as f:
        reader = csv.reader(f)
        # Skip header row
        header = next(reader)
        
        # Identify indices of columns to exclude
        exclude_columns = ['number', 'datetime_start', 'datetime_complete', 'duration', 'system_attrs_fixed_params', 'state']
        exclude_indices = [i for i, col_name in enumerate(header) if col_name in exclude_columns]
        
        # Filter out unwanted columns from each row
        filtered_rows = []
        for row in reader:
            if row and row[0] != "" and row[1] != "":  # Ensure row is not empty
                filtered_row = [col for i, col in enumerate(row) if i not in exclude_indices]
                filtered_rows.append(filtered_row)
        # Convert accuracy and flops values to float
        for row in filtered_rows:
            row[0] = float(row[0])  # Convert accuracy to float
            row[1] = float(row[1])  # Convert flops to float
        return filtered_rows
    
def image_get_baseline_acc_flops(model, test_loader):
    layer_num = len(model.get_conv_modules_dynamic_prune())
    model.my_eval([None] * layer_num)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    total_flops = model.gather_flops()
    return correct / total, total_flops / total


def find_pareto_frontier(rows, slices=5):
    frontier = set()
    for slice in range(slices):
        if len(frontier) == len(rows):
            print(f"all points included")
            break
        frontier_slice = []
        for row in rows:
            if tuple(row) in frontier:
                continue
            is_dominated = False
            for other_row in rows:
                if tuple(other_row) in frontier:
                    continue
                # Check if other_row dominates this row (higher accuracy and lower FLOPs)
                if other_row[0] >= row[0] and other_row[1] <= row[1] and (other_row[0] > row[0] or other_row[1] < row[1]):
                    is_dominated = True
                    break
            if not is_dominated:
                frontier_slice.append(row)
        for row in frontier_slice:
            frontier.add(tuple(row))
    return [list(row) for row in frontier]

def find_parato_frontier_test_trials(val_trials, test_trials):
    test_trials_dict = {}
    for row in test_trials:
        test_trials_dict[tuple(row[2:])] = (row[0], row[1])
    val_frontier = find_pareto_frontier(val_trials)
    test_frontier = [test_trials_dict[tuple(row[2:])] for row in val_frontier if tuple(row[2:]) in test_trials_dict]
    return test_frontier

def find_min_flops(test_stats_trials, test_thr_trials, baseline_acc, baseline_flops, fidelity_loss):
    min_flops_stats = float('inf')
    min_flops_thr = float('inf')
    
    # Check stats test trials
    for row in test_stats_trials:
        if row[0] >= baseline_acc - fidelity_loss:
            min_flops_stats = min(min_flops_stats, row[1])
    
    # Check threshold trials
    for row in test_thr_trials:
        if row[0] >= baseline_acc - fidelity_loss:
            min_flops_thr = min(min_flops_thr, row[1])

    min_flops = min(min_flops_stats, min_flops_thr)
    # Print results
    print(f"Largest FLOPs reduction (fidelity loss <= {fidelity_loss:.4f}): {(1-min_flops/baseline_flops)*100:.2f}%")

def plot_figure(val_stats_trials, test_stats_trials, val_thr_trials, test_thr_trials, baseline_acc, baseline_flops, name, metric_name, option="frontier", multiple_baseline=False):
    if option == "frontier":
        test_stats_trials = find_parato_frontier_test_trials(val_stats_trials, test_stats_trials)
        test_thr_trials = find_parato_frontier_test_trials(val_thr_trials, test_thr_trials)

    # Create a new figure
    plt.figure(figsize=(10, 4))
    
    # Plot the points for each trial type with different colors and markers
    marker_size = 400
    if not multiple_baseline:
        plt.scatter([row[1] / baseline_flops for row in test_stats_trials], [row[0] for row in test_stats_trials], 
                    color=stats_color, marker='o', label='Stats Test Confidence', alpha=0.7, s=marker_size)
        plt.scatter([row[1] / baseline_flops for row in test_thr_trials], [row[0] for row in test_thr_trials], 
                color=thr_color, marker='^', label='Threshold Confidence', alpha=0.7, s=marker_size)
    else:
        max_flops = max(baseline_flops)
        plt.scatter([row[1] / max_flops for row in test_stats_trials], [row[0] for row in test_stats_trials], 
                    color=stats_color, marker='o', label='Stats Test Confidence', alpha=0.7, s=marker_size)
        plt.scatter([row[1] / max_flops for row in test_thr_trials], [row[0] for row in test_thr_trials], 
                color=thr_color, marker='^', label='Threshold Confidence', alpha=0.7, s=marker_size)
    
    if not multiple_baseline:
        find_min_flops(test_stats_trials, test_thr_trials, baseline_acc, baseline_flops, 0.001)
        find_min_flops(test_stats_trials, test_thr_trials, baseline_acc, baseline_flops, 0.01)
        baseline_flops = 1.0
    else:
        baseline_flops = [x / max_flops for x in baseline_flops]
    # Plot the baseline point with a unique color and shape
    plt.scatter(baseline_flops, baseline_acc, color=baseline_color, marker='*', s=marker_size+200, 
                label='Baseline', zorder=5)

    custom_linestyles = {'loosely dotted': (0, (1, 3)), 'loosely dashed': (0, (5, 5))}

    axlinewidth = 6
    axlinealpha = 0.7
    # Add dotted lines for baseline accuracy and FLOPs
    if not multiple_baseline:
        plt.axhline(y=baseline_acc, color='black', linestyle=custom_linestyles['loosely dashed'], alpha=axlinealpha, 
                    label=f'Baseline {metric_name}', linewidth=axlinewidth)
        plt.axhline(y=baseline_acc-0.01, color='gray', linestyle=custom_linestyles['loosely dashed'], alpha=axlinealpha, 
                    label=f'Baseline {metric_name} - 1%', linewidth=axlinewidth)
        plt.axvline(x=baseline_flops, color='black', linestyle=custom_linestyles['loosely dashed'], alpha=axlinealpha, 
                    label='Baseline FLOPs', linewidth=axlinewidth)
    # else:
    #     for x, y in zip(baseline_flops, baseline_acc):
    #         plt.axhline(y=y, color='black', linestyle=custom_linestyles['loosely dashed'], alpha=axlinealpha, 
    #                     label=f'Baseline {metric_name}', linewidth=axlinewidth)
    #         plt.axvline(x=x, color='black', linestyle=custom_linestyles['loosely dashed'], alpha=axlinealpha, 
    #                     label='Baseline FLOPs', linewidth=axlinewidth)
    
    label_size = 30
    # Set labels and title
    plt.xlabel('normalized FLOPs', fontsize=label_size)
    plt.ylabel(f'{metric_name}', fontsize=label_size)
    # title_size = 40
    # plt.title('Model Performance: Accuracy vs. FLOPs', fontsize=title_size)
    
    # # Add legend
    # legend_size = 30
    # plt.legend(fontsize=legend_size)

    # Set tick label size
    tick_size = 25
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    # Format tick labels to show only two decimal places
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Save the figure
    if not os.path.exists("benchmark_plots"):
        os.makedirs("benchmark_plots")
    plt.savefig(f"benchmark_plots/{name}.pdf", dpi=300, bbox_inches='tight')

def get_trials(model_name):
    val_stats_study_name = model_name
    val_thr_study_name = f"{model_name}-ThresholdConfidence"
    test_stats_study_name = f"{val_stats_study_name}-test"
    test_thr_study_name = f"{val_thr_study_name}-test"

    val_stats_trials = load_trial_csv(val_stats_study_name)
    val_thr_trials = load_trial_csv(val_thr_study_name)
    test_stats_trials = load_trial_csv(test_stats_study_name)
    test_thr_trials = load_trial_csv(test_thr_study_name)
    return val_stats_trials, val_thr_trials, test_stats_trials, test_thr_trials

def plot_stats_thr(model_name, baseline_fidelity, baseline_flops, metric_name):
    val_stats_trials, val_thr_trials, test_stats_trials, test_thr_trials = get_trials(model_name)
    plot_figure(val_stats_trials, test_stats_trials, val_thr_trials, test_thr_trials, baseline_fidelity, baseline_flops, model_name, metric_name, option="frontier")

def plot_pruned_vgg():
    model_name, model = models.build_vgg11_bn_cifar10()
    model = model.to(DEVICE).eval()
    # baselines = [(0.7754,3362186.0), (0.7559, 3154146.0), (0.7456, 2965978.0)]
    # all_baseline_acc = [row[0] for row in baselines]
    # all_baseline_flops = [row[1] for row in baselines]
    all_baseline_acc = []
    all_baseline_flops = []
    all_test_stats_frontier_trials = []
    all_test_thr_frontier_trials = []
    _, _, test_loader = data.get_cifar10_data()
    for iter in range(77, 80):
        model = load_iterative_pruned_model(model_name, model, iter, 0.95)
        new_model_name = get_iterative_pruned_model_name(model_name, iter, 0.95)
        model = model.to(DEVICE).eval()
        baseline_acc, baseline_flops = image_get_baseline_acc_flops(model, test_loader)
        all_baseline_acc.append(baseline_acc)
        all_baseline_flops.append(baseline_flops)
        # baseline_acc, baseline_flops = baselines[iter-77]
        print(f"vgg11bn {iter} baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")
        val_stats_trials, val_thr_trials, test_stats_trials, test_thr_trials = get_trials(new_model_name)
        plot_figure(val_stats_trials, test_stats_trials, val_thr_trials, test_thr_trials, baseline_acc, baseline_flops, new_model_name, "accuracy", option="frontier")
        all_test_stats_frontier_trials.append(find_parato_frontier_test_trials(val_stats_trials, test_stats_trials))
        all_test_thr_frontier_trials.append(find_parato_frontier_test_trials(val_thr_trials, test_thr_trials))
    all_test_stats_frontier_trials = [item for sublist in all_test_stats_frontier_trials for item in sublist]
    all_test_thr_frontier_trials = [item for sublist in all_test_thr_frontier_trials for item in sublist]
    plot_figure(None, all_test_stats_frontier_trials, None, all_test_thr_frontier_trials, all_baseline_acc, all_baseline_flops, "pruned_vgg", "accuracy", option="all", multiple_baseline=True)

def plot_vgg():
    model_name, model = models.build_vgg11_bn_cifar10()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)
    _, _, test_loader = data.get_cifar10_data()
    baseline_acc, baseline_flops = image_get_baseline_acc_flops(model, test_loader)
    # baseline_acc, baseline_flops = 0.8811, 305896450.0
    print(f"vgg11bn baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")
    plot_stats_thr(model_name, baseline_acc, baseline_flops, "accuracy")

def plot_resnet():
    model_name, model = models.build_resnet18_cifar10()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)
    _, _, test_loader = data.get_cifar10_data()
    baseline_acc, baseline_flops = image_get_baseline_acc_flops(model, test_loader)
    # baseline_acc, baseline_flops = 0.9241, 1109706762.0
    print(f"resnet18bn baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")
    plot_stats_thr(model_name, baseline_acc, baseline_flops, "accuracy")

def plot_gcn():
    from gnn_utils import gnn_my_eval, get_gnn_dataset
    model_name, model = models.build_gnn()
    DEVICE = torch.device("cpu")
    dataset, train_loader, val_loader, test_loader, evaluator = get_gnn_dataset()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)
    res, baseline_flops = gnn_my_eval(model, test_loader, evaluator, [None] * (model.num_layer-1))
    baseline_rocauc = res[dataset.eval_metric]
    # baseline_rocauc, baseline_flops = 0.7780644662894225, 23674172.428884026
    print(f"gnn baseline_rocauc: {baseline_rocauc}, baseline_flops: {baseline_flops}")
    plot_stats_thr(model_name, baseline_rocauc, baseline_flops, "roc-auc")

def plot_opt():
    from opt_utils import eval_piqa
    from models.models import build_opt
    model_name, model, tokenizer = build_opt()
    model = model.to(DEVICE).eval()
    baseline_acc, baseline_flops = eval_piqa(model, tokenizer, [None] * model.get_num_dynamic_prune_layers(), split='validation')
    # baseline_acc, baseline_flops = 0.7105549510337323, 54107124046.27639
    print(f"opt baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")
    plot_stats_thr(model_name, baseline_acc, baseline_flops, "accuracy")

def main():
    plot_resnet()
    plot_vgg()
    plot_gcn()
    plot_opt()
    plot_pruned_vgg()

if __name__ == "__main__":
    main()