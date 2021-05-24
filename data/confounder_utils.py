import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.jigsaw_dataset import JigsawDataset

########################
####### SETTINGS #######
########################

confounder_settings = {
    "CelebA": {
        "constructor": CelebADataset
    },
    "CUB": {
        "constructor": CUBDataset
    },
    "MultiNLI": {
        "constructor": MultiNLIDataset
    },
    'jigsaw':{
        'constructor': JigsawDataset
    },
}


########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    if args.dataset != "jigsaw":
        full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data,
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv"
        )
    else: 
        full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data,
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv",
            batch_size=args.batch_size
        )
    
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
    if train:
        splits = ["train", "val", "test"]
    else:
        splits = ["test"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        ) for split in splits
    ]
    return dro_subsets
