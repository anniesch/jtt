import argparse
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def file_condition(file):
    return True
    return ("" in file) and ("" in file) and (f"epochs_{n_epochs}" in file)


def main(args):
    experiment_name = args.exp_name 
    original_path = args.metadata_path 
    n_epochs = args.n_epochs
    split= args.split
    original_df = pd.read_csv(original_path)


    groups = [ 
        'average_acc', 
        'male', 
        'female', 
        'christian', 
        'muslim', 
        'other_religion', 
        'black', 
        'white', 
        'LGBTQ'
        ]

    methods = [file for file in os.listdir(f"results/jigsaw/{experiment_name}/") if file_condition(file)]
    methods.sort()
    methods = [method for method in methods]

    # aug_path = f"results/jigsaw/{experiment_name}/downstream/metadata_aug.csv"
    aug_path = f"results/jigsaw/reprod_change_csv_but_nothing_else/metadata_aug.csv"
    aug_df = pd.read_csv(aug_path)

    val_df = aug_df[aug_df["split"] == split]
    val_df["label"] = (val_df["toxicity"] >= 0.5) + 0
    val_df = val_df.reset_index()

    val_df["average_acc"] = 1
    val_df["worst_group"] = 1
    val_df["LGBTQ"] = (val_df['transgender'] 
                    + val_df['homosexual_gay_or_lesbian'] 
                    + val_df['bisexual'] 
                    + val_df['other_sexual_orientation'] 
                    + val_df['other_gender']) > 0

    val_df["other_religion"] = (val_df['other_religion'] 
                                + val_df['hindu'] 
                                + val_df['jewish'] 
                                + val_df['atheist'] 
                                + val_df['buddhist']) > 0

    all_groups = {}
    split_word = "times" 
    for final_epoch in range(3):
        group_dict = {}
        for method in methods: 
            if os.path.exists(f"results/jigsaw/{experiment_name}/downstream/{method}/model_outputs"):
                extension = '_val' if split == "val" else ''
                output_path = f"results/jigsaw/{experiment_name}/downstream/{method}/model_outputs/split/output{extension}_None_{final_epoch}.csv"
                if os.path.exists(output_path):
                    output_df = pd.read_csv(output_path)
                    y_col = [col for col in output_df.columns if "y_true_None_epoch_" in col][0]
                    final_epoch = y_col.split("y_true_None_epoch_")[1].split("_")[0]
                    assert np.sum(val_df["label"] != output_df[f"y_true_None_epoch_{final_epoch}_val"]) == 0
                    group_accs = []
                    group_dict["_".join(method.split(split_word)[0].split("_")[2:-1])] = {}
                    group_dict["_".join(method.split(split_word)[0].split("_")[2:-1])]["epoch"] = final_epoch
                    for group in groups: 
                        for toxicity in range(20):
                            identifier = (val_df[group] == 1) & (val_df["label"] == toxicity)
                            if len(val_df[identifier]) > 0:
                                acc = (np.sum(val_df[identifier]["label"] ==  output_df[identifier][f"y_pred_None_epoch_{final_epoch}_val"])) / len(val_df[identifier])
                                if group != "average_acc": 
                                    group_accs.append(acc)
                                else: 
                                    avg_acc = (np.sum(val_df["label"] ==  output_df[f"y_pred_None_epoch_{final_epoch}_val"])) / len(val_df)
                                    group_dict["_".join(method.split(split_word)[0].split("_")[2:-1])][f"{group}"] = avg_acc 
                                group_dict["_".join(method.split(split_word)[0].split("_")[2:-1])][f"{group}_{toxicity}"] = acc
                    group_dict["_".join(method.split(split_word)[0].split("_")[2:-1])]["worst_group_acc"] = np.min(group_accs) 
            if group_dict != {}:
                all_groups[final_epoch] = group_dict


    pd.set_option('display.max_columns', None)

    results_df = pd.DataFrame()
    new_results_df = pd.DataFrame()
    for group_id, final_epoch in enumerate(all_groups.keys()):
        group_dict = all_groups[final_epoch]
        for row, method in enumerate(group_dict.keys()):
            for column in group_dict[method].keys():
                if row + group_id == 0:
                    results_df["method"] = method
                    results_df[column] = [group_dict[method][column]]
                else: 
                    new_results_df["method"] = method
                    new_results_df[column] = [group_dict[method][column]]

            if row + group_id != 0:
                results_df = results_df.append(new_results_df)

    print(results_df)     
    file_name = f"jigsaw_csv_results/{experiment_name}_{n_epochs}_{split}_jigsaw_group_results.csv"
    results_df.to_csv(file_name)
    print(f"saved {file_name}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="a name for the experiment directory",
        required=True
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="jigsaw/data/all_data_with_identities.csv",
        help="path to metadata",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="number of epochs in the run",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="split",
    )

    args = parser.parse_args()
    main(args)