This is the code repository for the paper "Exchangeability in Neural Network Architectures and its Application to Dynamic Pruning".

# Workflow to reproduce the experiments shown in the paper

The following steps reproduce all the results shown in our paper. It is simple to adapt the code to run only a subset of the experiments (e.g., GNN) quickly, by commenting out a few lines of irrelevant code in the `main` methods of the listed Python scripts.

## Train CNNs and GNN locally

Run the following commands:
```
python train_gnn.py
python train_model.py
```
The last command trains resnet18 and vgg11-bn on CIFAR10 dataset. It also prunes and finetunes the vgg11-bn model iteratively, storing the finetuned the model in each iteration. Pruning and finetuning take much more time than training the original models.

## Obtaining our customized `transformers` python module

In order to evaluate on OPT language model, we adapt the HuggingFace implementation of OPT by adding ExPrune optimization and code to record FLOPs.

The following steps install our customized `transformers` python module.
1. Download the adapted code at `https://anonymous.4open.science/r/transformers-DB8D/`.
2. Run `pip install .` (build from source).

We only changed `src/transformers/models/opt/modeling_opt.py` to insert `my_eval` and `gather_flops` methods.
Note that we do not train the OPT model locally but use the pretrained weights from HuggingFace.

## ExPrune hyperparamter optimization

Run `python tune_params.py`. It uses Optuna to find best ExPrune hyperparamter combinations (T for Threshold and alpha for StatsTest) on validation set, and evaluates all the found combinations on the test set.
The hyperparameter optimization for CNNs and GNNs completes in hours of time. The OPT experiment could take days to complete.

## Plot the results

Run `plot_benchmarks.py` to plot all the results as is in our paper.
