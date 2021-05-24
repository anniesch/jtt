import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
import argparse
import subprocess


def main(args):

    final_epoch = args.final_epoch
    dataset = args.dataset
    
    # CHANGE THESE FOLDERS
    exp_name = args.exp_name
    folder_name = args.folder_name
    data_dir = f"results/{args.dataset}/{exp_name}/{folder_name}/model_outputs/"
    if args.dataset == 'CelebA':
        metadata_path = "./celebA/data/metadata.csv"
    elif args.dataset == 'MultiNLI':
        metadata_path = "./multinli/data/metadata.csv"
    elif args.dataset == 'CUB':
        metadata_path = "./cub/data/waterbird_complete95_forest2water2/metadata.csv"
    elif args.dataset == "jigsaw":
        metadata_path = "./jigsaw/data/all_data_with_identities.csv"
    else: 
        raise NotImplementedError 
    
    # Load in train df and wrong points, this is the main part
    train_df = pd.read_csv(os.path.join(data_dir, f"output_train_epoch_{final_epoch}.csv"))
    train_df = train_df.sort_values(f"indices_None_epoch_{final_epoch}_val")
    train_df["wrong_1_times"] = (1.0 * (train_df[f"y_pred_None_epoch_{final_epoch}_val"] != train_df[f"y_true_None_epoch_{final_epoch}_val"])).apply(np.int64)
    print("Total wrong", np.sum(train_df['wrong_1_times']), "Total points", len(train_df))
    
    # Merge with original features (could be optional)
    original_df = pd.read_csv(metadata_path)
    original_train_df = original_df[original_df["split"] == 0]
    if dataset == "CelebA" or dataset == "jigsaw" or dataset == "MultiNLI":
        original_train_df = original_train_df.drop(['Unnamed: 0'], axis=1)

    merged_csv = original_train_df.join(train_df.set_index(f"indices_None_epoch_{final_epoch}_val"))
    if dataset == "CUB":
        merged_csv["spurious"] = merged_csv['y'] != merged_csv["place"]
    elif dataset == "CelebA":
        merged_csv = merged_csv.replace(-1, 0)
        assert 0 == np.sum(merged_csv[merged_csv["split"] == 0]["Blond_Hair"] != merged_csv[merged_csv["split"] == 0][f"y_true_None_epoch_{final_epoch}_val"])
        merged_csv["spurious"] = (merged_csv["Blond_Hair"] == merged_csv["Male"]) 
    elif dataset == "jigsaw":
        merged_csv["spurious"] = merged_csv["toxicity"] >= 0.5
    elif dataset == "MultiNLI":
        merged_csv["spurious"] = (
                (merged_csv["gold_label"] == 0)
                & (merged_csv["sentence2_has_negation"] == 0)
            ) | (
                (merged_csv["gold_label"] == 1)
                & (merged_csv["sentence2_has_negation"] == 1)
            )
    else: 
        raise NotImplementedError
    print("Number of spurious", np.sum(merged_csv['spurious']))
    
    # Make columns for our spurious and our nonspurious
    merged_csv["our_spurious"] = merged_csv["spurious"] & merged_csv["wrong_1_times"]
    merged_csv["our_nonspurious"] = (merged_csv["spurious"] == 0) & merged_csv["wrong_1_times"]
    print("Number of our spurious: ", np.sum(merged_csv["our_spurious"]))
    print("Number of our nonspurious:", np.sum(merged_csv["our_nonspurious"]))
    
    train_probs_df= merged_csv.fillna(0)
    
    # Output spurious recall and precision
    spur_precision = np.sum(
            (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
        ) / np.sum((merged_csv[f"wrong_1_times"] == 1))
    print("Spurious precision", spur_precision)
    spur_recall = np.sum(
        (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
    ) / np.sum((merged_csv["spurious"] == 1))
    print("Spurious recall", spur_recall)
    
    # Find confidence (just in case doing threshold)
    if dataset == "MultiNLI":
        probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1", f"pred_prob_None_epoch_{final_epoch}_val_2"]]), axis = 1)
        train_probs_df["probs_0"] = probs[:,0]
        train_probs_df["probs_1"] = probs[:,1]
        train_probs_df["probs_2"] = probs[:,2]
        train_probs_df["confidence"] = (train_probs_df['gold_label']==0) * train_probs_df["probs_0"] + (train_probs_df['gold_label']==1) * train_probs_df["probs_1"] + (train_probs_df['gold_label']==2) * train_probs_df["probs_2"]
    else:
        probs = softmax(np.array(train_probs_df[[f"pred_prob_None_epoch_{final_epoch}_val_0", f"pred_prob_None_epoch_{final_epoch}_val_1"]]), axis = 1)
        train_probs_df["probs_0"] = probs[:,0]
        train_probs_df["probs_1"] = probs[:,1]
        if dataset == 'CelebA':
            train_probs_df["confidence"] = train_probs_df["Blond_Hair"] * train_probs_df["probs_1"] + (1 - train_probs_df["Blond_Hair"]) * train_probs_df["probs_0"]
        elif dataset == 'CUB':
            train_probs_df["confidence"] = train_probs_df["y"] * train_probs_df["probs_1"] + (1 - train_probs_df["y"]) * train_probs_df["probs_0"]
        elif dataset == 'jigsaw':
            train_probs_df["confidence"] = (train_probs_df["toxicity"] >= 0.5) * train_probs_df["probs_1"] + (train_probs_df["toxicity"] < 0.5)  * train_probs_df["probs_0"]
    
    train_probs_df[f"confidence_thres{args.conf_threshold}"] = (train_probs_df["confidence"] < args.conf_threshold).apply(np.int64)
    if dataset == 'CelebA':
        assert(np.sum(train_probs_df[f"confidence_thres{args.conf_threshold}"] != train_probs_df["wrong_1_times"]) == 0)
    
    # Save csv into new dir for the run, and generate downstream runs
    if not os.path.exists(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"):
        os.makedirs(f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}")
    root = f"results/{dataset}/{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"

    train_probs_df.to_csv(f"{root}/metadata_aug.csv")
    root = f"{exp_name}/train_downstream_{folder_name}/final_epoch{final_epoch}"
    
    sbatch_command = (
            f"python generate_downstream.py --exp_name {root} --lr {args.lr} --weight_decay {args.weight_decay} --method JTT --dataset {args.dataset} --aug_col {args.aug_col}" + (f" --batch_size {args.batch_size}" if args.batch_size else "")
        )
    print(sbatch_command)
    if args.deploy:
        subprocess.run(sbatch_command, check=True, shell=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="CelebA", help="CUB, CelebA, or MultiNLI"
    )
    parser.add_argument(
        "--final_epoch",
        type=int,
        default=5,
        help="last epoch in training",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--deploy", action="store_true", default=False)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--aug_col", type=str, default='wrong_1_times')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()
    main(args)
    