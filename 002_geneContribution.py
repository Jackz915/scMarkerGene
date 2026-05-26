import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from captum.attr import DeepLift
import numpy as np
import copy
import os,sys,time,datetime
import json
import math
import re
import argparse
import optuna
from collections import Counter
import random
from typing import Tuple, Dict, Optional
from functools import partial
import itertools
from sklearn.utils.class_weight import compute_class_weight


def infoLine(message, infoType="info"):
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)

    if infoType == "ERROR":
        sys.exit()
    sys.stdout.flush()


def parse_arguments():
    """Parse and validate command line arguments for scMarkerGene training"""
    parser = argparse.ArgumentParser(
        description="Version: 1.0\n"
        "Description: Quantifying gene relevance using neural networks.\n"
        "Documentation: https://geroes.zhaopage.com",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument("-i", "--input-dir",
                        dest="input_dir",
                        type=str,
                        required=True,
                        help="Path to input data directory")
    
    required.add_argument("-o", "--output-dir",
                        dest="output_dir",
                        type=str,
                        required=True,
                        help="Path to save output models and results")

    # Training Configuration
    training = parser.add_argument_group('Training Configuration')
    training.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        type=int,
                        default=1024,
                        help="Batch size for training")

    # Hyperparameter Ranges
    hyperparams = parser.add_argument_group('Hyperparameter Ranges')
    hyperparams.add_argument("--hidden-layers-range",
                           nargs=2,
                           type=int,
                           default=[1, 3],
                           metavar=("MIN", "MAX"),
                           help="Hidden layers range (min max)")
    hyperparams.add_argument("--dropout-range",
                           nargs=2,
                           type=float,
                           default=[0.2, 0.3],
                           metavar=("MIN", "MAX"),
                           help="Dropout rate range (min max)")
    
    hyperparams.add_argument("--hidden-units-range",
                           nargs=2,
                           type=int,
                           default=[256, 512],
                           metavar=("MIN", "MAX"),
                           help="Hidden units range (min max)")
    
    hyperparams.add_argument("--lr-range",
                           nargs=2,
                           type=float,
                           default=[1e-5, 1e-3],
                           metavar=("MIN", "MAX"),
                           help="Learning rate range (min max)")

    # Data Processing
    data = parser.add_argument_group('Data Processing')
    data.add_argument("--noise-sigma",
                    type=float,
                    default=1.0,
                    help="Gaussian noise standard deviation")

    data.add_argument("--strategy", choices=["upsample", "weights", "none"], default="weights",  
                        help="Strategy for handling class imbalance. "
                         "Options: upsample (simple oversampling), "
                         "weights (class-weighted loss), "
                         "none (no balancing).")

    # Optuna Configuration
    optuna = parser.add_argument_group('Optuna Configuration')
    optuna.add_argument("-e", "--epochs",
                        dest="max_epochs",
                        type=int,
                        default=50,
                        help="Maximum training epochs")
    
    optuna.add_argument("--global_trials",
                      type=int,
                      default=30,
                      help="Number of global optimization trials")
    
    optuna.add_argument("--optuna_storage",
                      type=str,
                      default="sqlite:///db.sqlite3",
                      help="Optuna storage URL")
    
    optuna.add_argument("--study_name",
                      type=str,
                      default="scMarkerGene",
                      help="Optuna study identifier")

    # Refinement Configuration
    refine = parser.add_argument_group('Refinement Configuration')
    refine.add_argument("--refine_lr_num",
                        type=int,
                        default=5,
                        help="Number of candidate learning rates to test during refinement.")

    refine.add_argument("--refine_dropout_num",
                        type=int,
                        default=3,
                        help="Number of candidate dropout rates to test during refinement.")
    
    refine.add_argument("--refine_lr_ratio",
                        type=float,
                        default=0.3,
                        help='Relative range (+/- ratio) around best learning rate to generate candidates. \
                        For example, 0.3 means ±30%% around the best lr.')
    
    refine.add_argument("--refine_dropout_ratio",
                        type=float,
                        default=0.1,
                        help='Relative range (+/- ratio) around best dropout rate to generate candidates. \
                        For example, 0.1 means ±10%% around the best dropout.')

    args = parser.parse_args()

    return args


def readMeta(file_path):
    with open(file_path) as f:
        return {
            key: int(value) if value.isdigit() else value
            for line in f
            if line.strip() and not line.startswith('#')
            for key, value in [line.strip().split(maxsplit=1)]
        }

    
def readRawData(
    filepath: str,
    doUpsample: bool = False,
    target_samples_per_class: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data_list, label_list = [], []
    with open(filepath, "rt") as fi:
        for line in fi:
            parts = line.rstrip().split("\t")
            label = int(parts[1])
            features = np.array([float(x) for x in parts[2].split("|")], dtype=np.float32)
            data_list.append(features)
            label_list.append(label)
    
    original_data = np.stack(data_list)
    original_labels = np.array(label_list)
    
    class_dist = Counter(original_labels)
    
    original_dict = {
        'data': original_data,
        'labels': original_labels,
        'class_distribution': dict(class_dist)
    }
    
    upsampled_dict = None
    if doUpsample:
        target_samples = target_samples_per_class or max(class_dist.values())
        
        resampled_data, resampled_labels = [], []
        new_dist = Counter()
        
        for class_label in class_dist:
            class_mask = (original_labels == class_label)
            class_data = original_data[class_mask]
            current_count = len(class_data)
            new_dist[class_label] = target_samples
            
            if current_count < target_samples:
                repeat_times = target_samples // current_count
                remainder = target_samples % current_count
                
                resampled_data.append(np.tile(class_data, (repeat_times, 1)))
                resampled_labels.append(np.repeat(class_label, repeat_times * current_count))
                
                if remainder > 0:
                    indices = np.random.choice(range(current_count), remainder)
                    resampled_data.append(class_data[indices])
                    resampled_labels.append(np.full(remainder, class_label))
            else:
                resampled_data.append(class_data)
                resampled_labels.append(original_labels[class_mask])
        
        upsampled_data = np.concatenate(resampled_data)
        upsampled_labels = np.concatenate(resampled_labels)
        
        shuffle_idx = np.random.permutation(len(upsampled_labels))
        upsampled_data = upsampled_data[shuffle_idx]
        upsampled_labels = upsampled_labels[shuffle_idx]
        
        upsampled_dict = {
            'data': upsampled_data,
            'labels': upsampled_labels,
            'class_distribution': dict(new_dist)
        }

    return (original_dict, upsampled_dict) if doUpsample else (original_dict.copy(), None)


class loadedDataset(Dataset):
    def __init__(self, data_dict, noise_sigma=None):
        self.noise_sigma = noise_sigma        
        self.data, self.labels = data_dict['data'], data_dict['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.noise_sigma:
            data = data + np.random.normal(0, self.noise_sigma, size=data.shape).astype(np.float32)
        
        return torch.FloatTensor(data), torch.LongTensor([label])

        
def get_Dataloader(data_dict, batch_size, noise_sigma=None, shuffle=False, drop_last=False):
    dataset = loadedDataset(
        data_dict,
        noise_sigma=noise_sigma
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=4,  
        # pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return data_loader


def Net(args, trial=None, params=None):
    if trial is None and params is None:
        raise ValueError("Either `trial` or `params` must be provided.")

    if params is not None:
        n_layers = params["n_layers"]
        dropout_rate = params["dropout"]
    else:
        n_layers = trial.suggest_int("n_layers", 
                                     args.hidden_layers_range[0], 
                                     args.hidden_layers_range[1])
        dropout_rate = trial.suggest_float("dropout", 
                                           args.dropout_range[0], 
                                           args.dropout_range[1])

    layers = []
    in_features = args.gene_num

    for i in range(n_layers):
        if params is not None:
            out_features = params[f"n_units_l{i}"]
        else:
            out_features = trial.suggest_int(f"n_units_l{i}", 
                                             args.hidden_units_range[0], 
                                             args.hidden_units_range[1])
            
        layers.extend([
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        in_features = out_features

    layers.append(nn.Linear(in_features, args.class_num))
    return nn.Sequential(*layers)

    
def objective(trial, args, train_loader, test_loader):
    model = Net(args, trial).to(args.device)
    criterion = nn.CrossEntropyLoss()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])  
    lr = trial.suggest_float("lr", args.lr_range[0], args.lr_range[1], log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(args.max_epochs):
        train_loss, train_acc, val_loss, val_acc = train_validate_one_epoch(
            model, train_loader, test_loader, optimizer, criterion, args.device)

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc
    

def train_validate_one_epoch(model, train_loader, test_loader, optimizer, criterion, device):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for data, label in train_loader:
        data, label = data.to(device), label.squeeze().long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        train_correct += (output.argmax(1) == label).sum().item()
        train_total += label.size(0)
    avg_train_loss = train_loss / train_total
    avg_train_accuracy = train_correct / train_total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.squeeze().long().to(device)
            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
            val_correct += (output.argmax(1) == label).sum().item()
            val_total += label.size(0)
    avg_val_loss = val_loss / val_total
    avg_val_accuracy = val_correct / val_total

    return avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy


def inference(model, infer_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss() 
    
    loss_list = []
    predicted_list = []
    label_list = []
    digit_list = [] 
        
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(infer_loader):
            data = data.to(device)
            label = label.view(-1).long().to(device)
            
            digit = model(data)
            loss = criterion(digit, label)
            
            loss_list.append(loss.item())  
            
            predicted = digit.argmax(dim=1)  
            
            predicted_list.append(predicted.cpu().numpy())
            label_list.append(label.cpu().numpy())
            
            digit_prob = F.softmax(digit, dim=1).cpu().numpy() 
            digit_list.append(digit_prob)  
    
        predicted_save = np.concatenate(predicted_list) 
        label_save = np.concatenate(label_list)         
        digit_save = np.concatenate(digit_list, axis=0) 
    
        correct = (predicted_save == label_save).sum()
        total = len(label_save)
        avg_loss = np.mean(loss_list)

    return label_save, predicted_save, round(avg_loss, 4), correct, total, digit_save


def warmup(model, train_loader, lr, device, total_steps=30):
    model.train()
    criterion = nn.CrossEntropyLoss()

    def lr_lambda(step):
        return step / total_steps if step < total_steps else 1.0

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr * 0.01,  
                                 weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # print("total warmup steps:", total_steps)

    step = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        if step >= total_steps:
            break

        data, label = data.to(device), label.squeeze().long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1

            
def check_early_stopping(chk_metric, best_metric, best_count, patience=5):
    if chk_metric < best_metric:
        return chk_metric, 0, True  # new best
    else:
        best_count += 1
        if best_count > patience:
            return best_metric, best_count, False
        else:
            return best_metric, best_count, None  # continue


def explain(args, model, param_index, pool_loader, universal_dict):
    model.eval()
    attributions = []
    all_labels = []
    explainer = DeepLift(model, multiply_by_inputs=False)
    base_data = torch.FloatTensor(universal_dict['data'][:1]).to(args.device)
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(pool_loader):
            data, label = data.to(args.device), label.squeeze().long().to(args.device)
            att = explainer.attribute(data, target=label, baselines=base_data)
            att = att.cpu().numpy().squeeze(0) 
            data = data.cpu().numpy().squeeze(0)
            attributions.append(att * data)  
            all_labels.append(label.item())
            
    attributions = np.array(attributions)
    # att_zscore = zscore(attributions, axis=0)
    
    outfile = f"{args.output_dir}/explain_dir/model_{param_index}_explanation.dat"
    with open(outfile, "w") as fo:
        for idx, (label, att) in enumerate(zip(all_labels, attributions)):
            att_str = "|".join([f"{x:.6f}" for x in att])
            fo.write(f"{label}\t{att_str}\n")

    return attributions
    
def save_performance(args, model, param_index, train_loader, test_loader, pool_loader):
    model = model.to(args.device)

    results = {
        'train': inference(model, train_loader, args.device),
        'test': inference(model, test_loader, args.device),
        'pool': inference(model, pool_loader, args.device)
    }

    for name, (labels, preds, loss, correct, total, digit) in results.items():
        # output data # label_save,predicted_save,test_loss,correct,total,digit_save
        labelList = list(labels)
        predList = list(preds)
        digitList = [list(k) for k in digit]
        
        with open(f"{args.output_dir}/predict_dir/model_{param_index}_{name}.tab", "w") as f:
            f.write("label\tprediction\tscore\n")
            for i in range( len( labelList ) ):
                outline = str(int( labelList[i] )) + "\t" + str(int( predList[i] ))+ "\t" + "|".join( [str(k) for k in digitList[i] ] )
                f.write( outline + "\n" )
            
def generate_param_combinations(args, global_best_params):
    lr_center = global_best_params["lr"]
    lr_candidates = np.linspace(lr_center * (1 - args.refine_lr_ratio), 
                                lr_center * (1 + args.refine_lr_ratio), 
                                args.refine_lr_num)

    dropout_center = global_best_params["dropout"]
    dropout_candidates = np.linspace(dropout_center * (1 - args.refine_dropout_ratio), 
                                     dropout_center * (1 + args.refine_dropout_ratio), 
                                     args.refine_dropout_num)
    
    dropout_candidates = np.clip(dropout_candidates, 0.0, 1.0)
    param_combinations = list(itertools.product(lr_candidates, dropout_candidates))

    final_param_list = []
    for lr, dropout in param_combinations:
        params = global_best_params.copy()
        params["lr"] = lr
        params["dropout"] = dropout
        final_param_list.append(params)

    return final_param_list


def calculate_class_weights(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float)

    
if __name__ == "__main__":
    args = parse_arguments()
        
    # read meta
    meta = readMeta(f"{args.input_dir}/meta.dat")
    for key, value in meta.items():
        setattr(args, f"{key}_num", value)  
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    os.system("mkdir -p " + f"{args.output_dir}/saved_models")
    os.system("mkdir -p " + f"{args.output_dir}/predict_dir")
    os.system("mkdir -p " + f"{args.output_dir}/explain_dir")

    infoLine("------------- Report parameters -----------------")
    for key, value in vars(args).items():
        print(f"├── {key}: {value}")
    
    # Keep reference to original training data
    infoLine("------------- Loading Raw Data -----------------")
    # First read original training data (no processing)
    train_original_raw, _ = readRawData(f"{args.input_dir}/train.dat", doUpsample=False)
    train_original_loader = get_Dataloader(
        train_original_raw,
        args.batch_size,
        args.noise_sigma,
        shuffle=False  # No shuffle needed for evaluation
    )
    
    # Load training data according to strategy
    if args.strategy == "upsample":
        infoLine("Using upsampling strategy...")
        # Read data with upsampling
        train_original, train_upsampled = readRawData(
            f"{args.input_dir}/train.dat", 
            doUpsample=True,
            target_samples_per_class=None
        )
        train_data_for_loader = train_upsampled
        class_weights = None
        
    elif args.strategy == "weights":
        infoLine("Using class weight loss strategy...")
        train_labels = train_original_raw['labels']
        class_weights = calculate_class_weights(train_labels)
        infoLine(f"Class weights: {class_weights}")
        train_data_for_loader = train_original_raw
        
    elif args.strategy == "none":
        infoLine("Using no balancing strategy...")
        train_data_for_loader = train_original_raw
        class_weights = None
        
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Create training data loader (may contain upsampled data)
    train_loader = get_Dataloader(
        train_data_for_loader,
        args.batch_size, 
        args.noise_sigma, 
        shuffle=True
    )

    # Read other datasets
    test_original, _ = readRawData(f"{args.input_dir}/test.dat", doUpsample=False)
    pool_original, _ = readRawData(f"{args.input_dir}/pool.dat", doUpsample=False)
    universal_original, _ = readRawData(f"{args.input_dir}/universal.dat", doUpsample=False)
    
    test_loader = get_Dataloader(
        test_original,
        args.batch_size
    )
    pool_loader = get_Dataloader(
        pool_original, 1
    )

    infoLine("------------- Starting Global Search -----------------")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_name = f"{args.study_name}_global_{timestamp}"
    global_study = optuna.create_study(
        directions=["maximize"],
        storage=args.optuna_storage,
        study_name=global_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=20, interval_steps=5
        )
    )
    
    # Pass class weights to objective function
    objective_with_args = partial(
        objective,
        args=args,
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    global_study.optimize(objective_with_args, n_trials=args.global_trials, show_progress_bar=True)
    global_best_params = global_study.best_trials[0].params
    
    infoLine("------------- Starting Refinement -----------------")
    # Generate params list
    final_param_list = generate_param_combinations(args, global_best_params)
    with open(f"{args.output_dir}/final_param_list.json", "w") as f:
        json.dump(final_param_list, f, indent=4)

    for param_index, params in enumerate(final_param_list):
        infoLine(f"model {param_index} training")
        print(params)
        
        model = Net(args, params=params).to(args.device)
        warmup(model, train_loader, params['lr'], args.device)
    
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Select loss function according to strategy
        if args.strategy == "weights" and class_weights is not None:
            infoLine(f"Using weighted loss function, weights: {class_weights}")
            class_weights_device = class_weights.to(args.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_device)
        else:
            criterion = nn.CrossEntropyLoss()
    
        best_metric = 1e9
        best_count = 0
        for epoch in range(10000):
            train_loss, train_acc, val_loss, val_acc = train_validate_one_epoch(
                model,
                train_loader,
                test_loader,
                optimizer,
                criterion,
                args.device
            )
    
            current_lr = optimizer.param_groups[0]['lr']
            infoLine(f"epoch:{epoch+1}\tloss_train:{train_loss:.4f}\tloss_test:{val_loss:.4f}"
                     f"\taccuracy_train:{train_acc*100:.2f}%\taccuracy_test:{val_acc*100:.2f}%"
                     f"\tcurrent_lr:{current_lr:.4f}")
    
            best_metric, best_count, status = check_early_stopping(train_loss, best_metric, best_count)
            if status is True:
                torch.save(
                    model.state_dict(),
                    f"{args.output_dir}/saved_models/candidate_{param_index}.pt"
                )
            elif status is False:
                infoLine(f"No improvement for {best_count} epochs. Stopping.")
                break
    
            scheduler.step(train_loss)
    
        # Key modification: evaluate performance using original training data
        infoLine(f"Evaluating model {param_index} performance using original training data...")
        save_performance(
            args, 
            model, 
            param_index, 
            train_original_loader,  # Use original training data, not upsampled
            test_loader, 
            pool_loader
        )
        explain(args, model, param_index, pool_loader, universal_original)
            
        infoLine("Done!")
