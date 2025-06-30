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

def trial_in_study(existing_param_set, params):
    return tuple(params.items()) in existing_param_set

def tune_cifar10(sampler, model_name, model, prune=False, confidence_class=StatsTestConfidence):
    prune_batch_num = 5
    max_acc_loss = 0.02

    def evaluate_accuracy(model, dataloader, prune_batch_num=None, prune_threshold=0):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if prune_batch_num is not None and i == prune_batch_num:
                    if correct / total < prune_threshold:
                        print(f"pruned, partial accuracy: {correct}/{total} = {correct / total}")
                        raise optuna.TrialPruned()
                del outputs
        return correct / total, total

    def get_num_layers(model):
        # return the number of layers that can be dynamically pruned in the model
        return len(model.get_conv_modules_dynamic_prune())

    def eval_acc_flops(val_loader, model, baseline_acc, confidence_class, param_range, trial):
        layer_num = get_num_layers(model)
        params = [trial.suggest_float(f"param_{i}", *param_range) for i in range(layer_num)]
        confidences = [confidence_class(params[i]) for i in range(layer_num)]
        if prune:
            gain_ratio_threshold_range = (0.1, 0.5)
            gain_ratio_thresholds = [trial.suggest_float(f"gain_ratio_threshold_{i}", *gain_ratio_threshold_range) for i in range(layer_num)]
            model.my_eval(confidences, gain_ratio_thresholds)
        else:
            model.my_eval(confidences)
        acc, total = evaluate_accuracy(model, val_loader, prune_batch_num=prune_batch_num, prune_threshold = baseline_acc - max_acc_loss)
        flops = model.gather_flops() / total
        print(f"acc: {acc}, flops: {flops}")
        return acc, flops

    def get_baseline_acc_flops(model, data_loader):
        model.my_eval([None] * get_num_layers(model))
        acc, total = evaluate_accuracy(model, data_loader)
        flops = model.gather_flops() / total
        return acc, flops
    
    train_loader, val_loader, test_loader = data.get_cifar10_data()

    if confidence_class == StatsTestConfidence:
        study_name = model_name
    else:
        study_name = f"{model_name}-{confidence_class.__name__}"

    baseline_acc, baseline_flops = get_baseline_acc_flops(model, val_loader)
    print(f"baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")

    def objective(trial):
        if confidence_class == StatsTestConfidence:
            param_range = (0.0, 0.5)
        else:
            param_range = (-30.0, 0.0)
        acc, flops = eval_acc_flops(val_loader, model, baseline_acc, confidence_class, param_range, trial)
        return acc, flops

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage="sqlite:///optuna.db",
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )

    def plot_study_on_test_set():
        # optuna.delete_study(study_name='cifar10_vgg11_bn-prune-rate-0.95-test', storage="sqlite:///optuna.db")
        # exit()
        study_test = optuna.create_study(
            directions=["maximize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=f"{study_name}-test",
            load_if_exists=True
        )
        existing_param_set = set()
        for trial in study_test.trials:
            existing_param_set.add(tuple(trial.params.items()))
        if confidence_class == StatsTestConfidence:
            param_range = (0.0, 0.5)
        else:
            param_range = (-30.0, 0.0)
        max_acc_loss_on_val = 0.02
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.PRUNED or trial.values is None:
                continue
            acc, flops = trial.values
            if acc < baseline_acc - max_acc_loss_on_val:
                continue
            if trial_in_study(existing_param_set, trial.params):
                # print(f"trial {trial.params} already in study_test")
                continue
            acc, flops = eval_acc_flops(test_loader, model, 0, confidence_class, param_range, trial)
            trial = optuna.trial.create_trial(
                values=[acc, flops],
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs
            )
            study_test.add_trial(trial)

        baseline_test_acc, baseline_test_flops = get_baseline_acc_flops(model, test_loader)
        print(f"baseline_test_acc: {baseline_test_acc}, baseline_test_flops: {baseline_test_flops}")
        plot_study(study_test, baseline_test_acc, "accuracy", baseline_test_flops, include_dominated_trials=True)

    num_trials = 2000
    existing_trials = len(study.trials)
    existing_trials_set = set()
    for trial in study.trials:
        existing_trials_set.add(tuple(trial.params.items()))
    if existing_trials < num_trials:
        layer_num = get_num_layers(model)
        print(f"layer_num: {layer_num}")
        for i in range(10):
            if confidence_class == StatsTestConfidence:
                point = {f"param_{j}": i * 0.01 for j in range(layer_num)}
            else:
                point = {f"param_{j}": -i * 3 for j in range(layer_num)}
            if prune:
                point.update({f"gain_ratio_threshold_{j}": 0.1 for j in range(layer_num)})
            if not trial_in_study(existing_trials_set, point):
                study.enqueue_trial(point)
        study.optimize(objective, n_trials=num_trials - existing_trials)
    plot_study(study, baseline_acc, "accuracy", baseline_flops)
    plot_study_on_test_set()

def tune_gnn(sampler, confidence_class=StatsTestConfidence):
    DEVICE = torch.device("cpu")
    from gnn_utils import gnn_my_eval, get_gnn_dataset

    dataset, train_loader, val_loader, test_loader, evaluator = get_gnn_dataset()

    def eval_acc_flops(data_loader, model, confidence_class, param_range, dataset, evaluator, trial):
        layer_num = model.num_layer - 1
        params = [trial.suggest_float(f"param_{i}", *param_range) for i in range(layer_num)]
        confidences = [None] * layer_num
        for i in range(layer_num):
            if confidence_class == StatsTestConfidence:
                if params[i] > 0.01:
                    confidences[i] = confidence_class(params[i])
            else:
                confidences[i] = confidence_class(params[i])
        res, flops = gnn_my_eval(model, data_loader, evaluator, confidences)
        rocauc = res[dataset.eval_metric]
        print(f"rocauc: {rocauc}, flops: {flops}")
        return rocauc, flops

    model_name, model = models.build_gnn()
    if confidence_class == StatsTestConfidence:
        study_name = model_name
    else:
        study_name = f"{model_name}-{confidence_class.__name__}"
    # optuna.delete_study(study_name=study_name, storage="sqlite:///optuna.db")
    # exit()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)

    res, baseline_flops = gnn_my_eval(model, val_loader, evaluator, [None] * (model.num_layer-1))
    baseline_rocauc = res[dataset.eval_metric]
    print(f"baseline_rocauc: {baseline_rocauc}, baseline_flops: {baseline_flops}")

    def objective(trial):
        if confidence_class == StatsTestConfidence:
            param_range = (0.0, 0.5)
        else:
            param_range = (-1.0, 0.0)
        rocauc, flops = eval_acc_flops(val_loader, model, confidence_class, param_range, dataset, evaluator, trial)
        return rocauc, flops
    
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage="sqlite:///optuna.db",
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )

    def plot_study_on_test_set():
        # optuna.delete_study(study_name='gnn-test', storage="sqlite:///optuna.db")
        # exit()
        study_test = optuna.create_study(
            directions=["maximize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=f"{study_name}-test",
            load_if_exists=True
        )
        if confidence_class == StatsTestConfidence:
            param_range = (0.0, 0.5)
        else:
            param_range = (-1.0, 0.0)
        max_rocauc_loss_on_val = 0.02
        existing_param_set = set()
        for trial in study_test.trials:
            existing_param_set.add(tuple(trial.params.items()))
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.PRUNED or trial.values is None:
                continue
            rocauc, flops = trial.values
            if rocauc < baseline_rocauc - max_rocauc_loss_on_val:
                continue
            if trial_in_study(existing_param_set, trial.params):
                # print(f"trial {trial.params} already in study_test")
                continue
            rocauc, flops = eval_acc_flops(test_loader, model, confidence_class, param_range, dataset, evaluator, trial)
            trial = optuna.trial.create_trial(
                values=[rocauc, flops],
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs
            )
            study_test.add_trial(trial)
        res, baseline_test_flops = gnn_my_eval(model, test_loader, evaluator, [None] * (model.num_layer-1))
        baseline_test_rocauc = res[dataset.eval_metric]
        print(f"baseline_test_rocauc: {baseline_test_rocauc}, baseline_test_flops: {baseline_test_flops}")
        plot_study(study_test, baseline_test_rocauc, "rocauc", baseline_test_flops, include_dominated_trials=True)

    n_trials = 2000
    existing_trials = len(study.trials)
    if existing_trials < n_trials:
        # add the initial trials to the study
        existing_trial_set = set()
        for trial in study.trials:
            existing_trial_set.add(tuple(trial.params.items()))
        layer_num = model.num_layer - 1
        print(f"layer_num: {layer_num}")
        if confidence_class == StatsTestConfidence:
            for i in range(10):
                point = {f"param_{j}": i * 0.01 for j in range(layer_num)}
                point["param_0"] = 0.0
                if not trial_in_study(existing_trial_set, point):
                    study.enqueue_trial(point)
        else:
            for i in range(11):
                point = {f"param_{j}": -i * 0.1 for j in range(layer_num)}
                point["param_0"] = -1
                if not trial_in_study(existing_trial_set, point):
                    study.enqueue_trial(point)
        study.optimize(objective, n_trials=n_trials - existing_trials)
    plot_study(study, baseline_rocauc, "rocauc", baseline_flops)
    plot_study_on_test_set()

def tune_opt(sampler, confidence_class=StatsTestConfidence):
    from opt_utils import eval_piqa
    from models.models import build_opt

    max_acc_loss = 0.05
    prune_checkpoint = 300

    model_name, model, tokenizer = build_opt()
    model.to(DEVICE)
    model.eval()

    def get_num_layers(model):
        # return the number of layers that can be dynamically pruned in the model
        return model.get_num_dynamic_prune_layers()

    def get_baseline_acc_flops(model, tokenizer, split):
        acc, flops = eval_piqa(model, tokenizer, [None] * get_num_layers(model), split=split)
        return acc, flops

    baseline_acc, baseline_flops = get_baseline_acc_flops(model, tokenizer, split="train")
    print(f"baseline_acc: {baseline_acc}, baseline_flops: {baseline_flops}")

    def eval_acc_flops(model, tokenizer, split, baseline_acc, confidence_class, param_range, trial):
        try:
            layer_num = get_num_layers(model)
            # Suggest boolean variables for each layer to determine if early termination is used
            use_early_term = [trial.suggest_categorical(f"use_early_term_{i}", [True, False]) for i in range(layer_num)]
            # For layers that use early termination, suggest alpha values
            alphas = [trial.suggest_float(f"param_{i}", *param_range) if use_early_term[i] else 0.0 for i in range(layer_num)]
            # Set confidence to None for layers that don't use early termination
            confidences = [confidence_class(alpha) if use_early_term[i] else None for i, alpha in enumerate(alphas)]
            acc, flops = eval_piqa(model, tokenizer, confidences, split=split, prune_checkpoint=prune_checkpoint, acc_lower_bound=baseline_acc - max_acc_loss)
            print(f"acc: {acc}, flops: {flops}")
            # Ensure both values are numeric
            if not isinstance(acc, (int, float)) or not isinstance(flops, (int, float)):
                print(f"acc: {acc}, flops: {flops}")
                raise ValueError("Accuracy or FLOPs is not a numeric value")
            return acc, flops
        except Exception as e:
            print(f"Error in eval_acc_flops: {str(e)}")
            raise

    def objective(trial):
        try:
            if confidence_class == StatsTestConfidence:
                param_range = (0.0, 0.5)
            else:
                param_range = (-0.005, 0.0)
            acc, flops = eval_acc_flops(model, tokenizer, 'train', baseline_acc, confidence_class, param_range, trial)
            # Ensure both values are floats
            return float(acc), float(flops)
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            raise optuna.TrialPruned()

    if confidence_class == StatsTestConfidence:
        study_name = model_name
    else:
        study_name = f"{model_name}-{confidence_class.__name__}"
    # optuna.delete_study(study_name=study_name, storage="sqlite:///optuna.db")
    # exit()
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage="sqlite:///optuna.db",
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )

    def plot_study_on_test_set():
        # optuna.delete_study(study_name=f"{study_name}-test", storage="sqlite:///optuna.db")
        # exit()
        study_test = optuna.create_study(
            directions=["maximize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=f"{study_name}-test",
            load_if_exists=True
        )
        baseline_test_acc, baseline_test_flops = get_baseline_acc_flops(model, tokenizer, split="validation")
        print(f"baseline_test_acc: {baseline_test_acc}, baseline_test_flops: {baseline_test_flops}")
        if confidence_class == StatsTestConfidence:
            param_range = (0.0, 0.5)
        else:
            param_range = (-0.005, 0.0)
        max_acc_loss_on_val = 0.02
        existing_param_set = set()
        for trial in study_test.trials:
            existing_param_set.add(tuple(trial.params.items()))
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.PRUNED or trial.values is None:
                continue
            acc, flops = trial.values
            if acc < baseline_acc - max_acc_loss_on_val:
                continue
            if trial_in_study(existing_param_set, trial.params):
                # print(f"trial {trial.params} already in study_test")
                continue
            print(f"trial {trial.params}")
            acc, flops = eval_acc_flops(model, tokenizer, 'validation', baseline_acc, confidence_class, param_range, trial)
            trial = optuna.trial.create_trial(
                values=[acc, flops],
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs
            )
            study_test.add_trial(trial)
        plot_study(study_test, baseline_test_acc, "Accuracy", baseline_test_flops, include_dominated_trials=True)

    n_trials = 2000
    existing_trials = len(study.trials)
    if existing_trials < n_trials:
        # add the initial trials to the study
        existing_trial_set = set()
        for trial in study.trials:
            existing_trial_set.add(tuple(trial.params.items()))
        layer_num = get_num_layers(model)
        print(f"layer_num: {layer_num}")
        base_point = {f"use_early_term_{i}": False for i in range(layer_num)}
        if not trial_in_study(existing_trial_set, base_point):
            study.enqueue_trial(base_point)
        param_val_trials = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] if confidence_class == StatsTestConfidence else [-0.005, -0.004, -0.003, -0.002, -0.001, 0]
        for i in range(layer_num):
            point = deepcopy(base_point)
            point[f"use_early_term_{i}"] = True
            for param_val in param_val_trials:
                point[f"param_{i}"] = param_val
                if not trial_in_study(existing_trial_set, point):
                    study.enqueue_trial(point)
        base_point = {f"use_early_term_{i}": True for i in range(layer_num)}
        param_val_trials = [0, 0.01, 0.02, 0.03, 0.04, 0.05] if confidence_class == StatsTestConfidence else [-0.005, -0.004, -0.003, -0.002, -0.001, 0]
        for param_val in param_val_trials:
            for i in range(layer_num):
                base_point[f"param_{i}"] = param_val
            if not trial_in_study(existing_trial_set, base_point):
                study.enqueue_trial(base_point)
        study.optimize(objective, n_trials=n_trials - existing_trials)
    
    plot_study(study, baseline_acc, "Accuracy", baseline_flops)
    plot_study_on_test_set()

def plot_study(study, baseline_perf, baseline_perf_name, baseline_flops, include_dominated_trials=False):
    if not os.path.exists("optuna_results"):
        os.makedirs("optuna_results")
    
    # Save the study trials dataframe to a CSV file
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"optuna_results/{study.study_name}_trials.csv", index=False)
    
    # Also save as pickle for easier loading later if needed
    with open(f"optuna_results/{study.study_name}_trials.pkl", "wb") as f:
        pickle.dump(trials_df, f)
    
    print(f"Saved trials dataframe to optuna_results/{study.study_name}_trials.csv")
    
    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=[baseline_perf_name, "flops"],
        axis_order=[1, 0],
        include_dominated_trials=include_dominated_trials
    )
    
    fig.add_hline(y=baseline_perf, line_dash="dash", line_color="red", annotation_text=f"baseline {baseline_perf_name}")
    fig.add_vline(x=baseline_flops, line_dash="dash", line_color="red", annotation_text="baseline FLOPs")
    # fig.show()
    fig.write_image(f"optuna_results/{study.study_name}.png")

def main():
    torch.manual_seed(123)

    # make the experiment reproducible
    if os.path.exists("optuna_sampler.pkl"):
        with open("optuna_sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
    else:
        sampler = optuna.samplers.TPESampler()
        with open("optuna_sampler.pkl", "wb") as f:
            pickle.dump(sampler, f)

    model_name, model = models.build_resnet18_cifar10()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)
    tune_cifar10(sampler, model_name, model)
    tune_cifar10(sampler, model_name, model, confidence_class=ThresholdConfidence)
    
    model_name, model = models.build_vgg11_bn_cifar10()
    model = model.to(DEVICE).eval()
    model = load_model(model_name, model)
    tune_cifar10(sampler, model_name, model)
    tune_cifar10(sampler, model_name, model, confidence_class=ThresholdConfidence)

    model_name, model = models.build_vgg11_bn_cifar10()
    model = model.to(DEVICE).eval()
    for iter in range(77, 80):
        model = load_iterative_pruned_model(model_name, model, iter, 0.95)
        new_model_name = get_iterative_pruned_model_name(model_name, iter, 0.95)
        tune_cifar10(sampler, new_model_name, model, prune=True)
        tune_cifar10(sampler, new_model_name, model, confidence_class=ThresholdConfidence, prune=True)
    
    tune_gnn(sampler)
    tune_gnn(sampler, confidence_class=ThresholdConfidence)

    tune_opt(sampler)
    tune_opt(sampler, confidence_class=ThresholdConfidence)

if __name__ == "__main__":
    main()