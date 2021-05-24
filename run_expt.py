import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
import wandb
from copy import deepcopy

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data import dro_dataset
from data import folds
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss
from train import train
from data.folds import Subset, ConcatDataset


def main(args):
    if args.wandb:
        wandb.init(project=f"{args.project_name}_{args.dataset}")
        wandb.config.update(args)

    # BERT-specific configs copied over from run_glue.py
    if (args.model.startswith("bert") and args.use_bert_params): 
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, "log.txt"), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == "confounder":
        train_data, val_data, test_data = prepare_data(
            args,
            train=True,
        )

    elif args.shift_type == "label_shift_step":
        raise NotImplementedError
        train_data, val_data = prepare_data(args, train=True)

    #########################################################################
    ###################### Prepare data for our method ######################
    #########################################################################

    # Should probably not be upweighting if folds are specified.
    assert not args.fold or not args.up_weight

    # Fold passed. Use it as train and valid.
    if args.fold:
        train_data, val_data = folds.get_fold(
            train_data,
            args.fold,
            cross_validation_ratio=(1 / args.num_folds_per_sweep),
            num_valid_per_point=args.num_sweeps,
            seed=args.seed,
        )

    if args.up_weight != 0:
        assert args.aug_col is not None
        # Get points that should be upsampled
        metadata_df = pd.read_csv(args.metadata_path)
        if args.dataset == "jigsaw":
            train_col = metadata_df[metadata_df["split"] == "train"]
        else:
            train_col = metadata_df[metadata_df["split"] == 0]
        aug_indices = np.where(train_col[args.aug_col] == 1)[0]
        print("len", len(train_col), len(aug_indices))
        if args.up_weight == -1:
            up_weight_factor = int(
                (len(train_col) - len(aug_indices)) / len(aug_indices)) - 1
        else:
            up_weight_factor = args.up_weight

        print(f"Up-weight factor: {up_weight_factor}")
        upsampled_points = Subset(train_data,
                                  list(aug_indices) * up_weight_factor)
        # Convert to DRODataset
        train_data = dro_dataset.DRODataset(
            ConcatDataset([train_data, upsampled_points]),
            process_item_fn=None,
            n_groups=train_data.n_groups,
            n_classes=train_data.n_classes,
            group_str_fn=train_data.group_str,
        )
    elif args.aug_col is not None:
        print("\n"*2 + "WARNING: aug_col is not being used." + "\n"*2)

    #########################################################################
    #########################################################################
    #########################################################################

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_loader = dro_dataset.get_loader(train_data,
                                          train=True,
                                          reweight_groups=args.reweight_groups,
                                          **loader_kwargs)

    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)

    if test_data is not None:
        test_loader = dro_dataset.get_loader(test_data,
                                             train=False,
                                             reweight_groups=None,
                                             **loader_kwargs)

    data = {}
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data

    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    model = get_model(
        model=args.model,
        pretrained=not args.train_from_scratch,
        resume=resume,
        n_classes=train_data.n_classes,
        dataset=args.dataset,
        log_dir=args.log_dir,
    )
    if args.wandb:
        wandb.watch(model)

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if resume:
        raise NotImplementedError  # Check this implementation.
        df = pd.read_csv(os.path.join(args.log_dir, "test.csv"))
        epoch_offset = df.loc[len(df) - 1, "epoch"] + 1
        logger.write(f"starting from epoch {epoch_offset}")
    else:
        epoch_offset = 0

    
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"train.csv"),
                                      train_data.n_groups,
                                      mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv"),
                                    val_data.n_groups,
                                    mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                     test_data.n_groups,
                                     mode=mode)
    train(
        model,
        criterion,
        data,
        logger,
        train_csv_logger,
        val_csv_logger,
        test_csv_logger,
        args,
        epoch_offset=epoch_offset,
        csv_name=args.fold,
        wandb=wandb if args.wandb else None,
    )

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()


def check_args(args):
    if args.shift_type == "confounder":
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith("label_shift"):
        assert args.minority_fraction
        assert args.imbalance_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        required=True)
    parser.add_argument("-s",
                        "--shift_type",
                        choices=shift_types,
                        required=True)
                        
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="spurious", help="wandb project name")
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    parser.add_argument("--up_weight", type=int, default=0)
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Objective
    parser.add_argument("--loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                         help=("Size param for CVaR joint DRO."
                               " Only used if loss_type is joint_dro"))
    parser.add_argument("--use_normalized_loss",
                        default=False,
                        action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")
    # Model
    parser.add_argument("--model",
                        choices=model_attributes.keys(),
                        default="resnet50")
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    # Optimization
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.7)

    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default="metadata.csv",
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument("--fold", default=None)
    # Our groups (upweighting/dro_ours)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="path to metadata csv",
    )
    parser.add_argument("--aug_col", default=None)

    args = parser.parse_args()
    
    if args.model.startswith("bert"): # and args.model != "bert": 
        if args.use_bert_params:
            print("\n"*5, f"Using bert params", "\n"*5)
        else: 
            print("\n"*5, f"WARNING, Using {args.model} without using BERT HYPER-PARAMS", "\n"*5)

    check_args(args)
    if args.metadata_csv_name != "metadata.csv":
        print("\n" * 2
              + f"WARNING: You are using '{args.metadata_csv_name}' instead of the default 'metadata.csv'."
              + "\n" * 2)
        
    main(args)
