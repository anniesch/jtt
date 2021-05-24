import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
join = os.path.join

def process_df_waterbird9(train_df, val_df, test_df, params):
    process_df(train_df, val_df, test_df, params)
    loss_metrics = []
    acc_metrics = []
    for group_idx in range(params["n_groups"]):
        loss_metrics.append(f"avg_loss_group:{group_idx}")
        acc_metrics.append(f"avg_acc_group:{group_idx}")

    ratio = params["n_train"] / np.sum(params["n_train"])
    val_df["avg_acc"] = val_df.loc[:, acc_metrics] @ ratio
    val_df["avg_loss"] = val_df.loc[:, loss_metrics] @ ratio
    test_df["avg_acc"] = test_df.loc[:, acc_metrics] @ ratio
    test_df["avg_loss"] = test_df.loc[:, loss_metrics] @ ratio


def sanitize_df(df):
    """
    Fix a results df for problems arising from resuming.
    """
    # Remove stray epoch/batches
    duplicates = df.duplicated(subset=["epoch", "batch"], keep="last")
    df = df.loc[~duplicates, :]
    df.index = np.arange(len(df))

    if np.sum(duplicates) > 0:
        print(
            f"Removed {np.sum(duplicates)} duplicates from epochs {np.unique(df.loc[duplicates, 'epoch'])}"
        )

    # Make sure epoch/batch is increasing monotonically
    prev_epoch = -1
    prev_batch = -1
    last_batch_in_epoch = -1
    for i in range(len(df)):
        try:
            epoch, batch = df.loc[i, ["epoch", "batch"]].astype(int)
        except:
            print(i, epoch, batch, len(df))
        assert ((prev_epoch == epoch) and
                (prev_batch < batch)) or ((prev_epoch == epoch - 1))
        if prev_epoch == epoch - 1:
            assert (last_batch_in_epoch == -1) or (last_batch_in_epoch
                                                   == prev_batch)
            last_batch_in_epoch = prev_batch
        prev_epoch = epoch
        prev_batch = batch

    return df


def get_accs_for_epoch_across_batches(df, epoch):
    n_groups = 1 + np.max([
        int(col.split(":")[1])
        for col in df.columns if col.startswith("avg_acc_group")
    ])

    indices = df["epoch"] == epoch

    accs = np.zeros(n_groups)
    total_counts = np.zeros(n_groups)
    correct_counts = np.zeros(n_groups)

    for i in np.where(indices)[0]:
        for group in range(n_groups):
            total_counts[group] += df.loc[
                i, f"processed_data_count_group:{group}"]
            correct_counts[group] += np.round(
                df.loc[i, f"avg_acc_group:{group}"] *
                df.loc[i, f"processed_data_count_group:{group}"])

    accs = correct_counts / total_counts
    robust_acc = np.min(accs)
    avg_acc = accs @ total_counts / np.sum(total_counts)
    return avg_acc, robust_acc



def print_accs(
    dfs,
    output_dir,
    params=None,
    epoch_to_eval=None,
    print_avg=False,
    output=True,
    splits=["train", "val", "test"],
    early_stop=True,
    print_groups = False,
):
    """
    Input: dictionary of dfs with keys 'val', 'test'
    This takes the minority group 'n' for calculating stdev,
    which is conservative.
    Since clean val/test acc for waterbirds is estimated from a val/test set with a different distribution, there's probably a bit more variability,
    but this is minor since the overall n is high.
    """
    for split in splits:
        assert split in dfs

    early_stopping_epoch = np.argmax(dfs["val"]["robust_acc"].values)

    epochs = []
    assert early_stop or (epoch_to_eval is not None)
    if early_stop:
        epochs += [("early stop at epoch", "early_stopping",
                    early_stopping_epoch)]
    if epoch_to_eval is not None:
        epochs += [("epoch", "epoch_to_eval", epoch_to_eval)]

    metrics = [("Val Robust Worst Group", "robust_acc")]
    if print_avg:
        metrics += [("Val Average Acc", "avg_acc")]
    if print_groups: 
        for i in range(group_count):
            metrics += [(f"group {i} acc", f"avg_acc_group:{i}")]

    results = {}
    for metric_str, metric in metrics:
        results[metric] = {}

        for split in splits:
            for epoch_print_str, epoch_save_str, epoch in epochs:
                if epoch not in dfs[split]["epoch"].values:
                    if output:
                        print(
                            f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch_to_eval}):               Not yet run"
                        )
                else:
                    if split == "train":
                        avg_acc, robust_acc = get_accs_for_epoch_across_batches(
                            dfs[split], epoch)
                        if metric == "avg_acc":
                            acc = avg_acc
                        elif metric == "robust_acc":
                            acc = robust_acc
                    else:
                        idx = np.where(dfs[split]["epoch"] == epoch)[0][
                            -1]  # Take the last batch in this epoch
                        acc = dfs[split].loc[idx, metric]

                    if split not in results[metric]:
                        results[metric][split] = {}

                    if params is None:
                        if output:
                            print(
                                f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch}): "
                                f"{acc*100:.1f}")
                            with open(output_dir + "/val_accuracies.txt",
                                      "a") as text_file:
                                print(
                                    f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch}): "
                                    f"{acc*100:.1f}",
                                    file=text_file,
                                )
                    else:
                        n_str = f"n_{split}"
                        minority_n = np.min(params[n_str])
                        total_n = np.sum(params[n_str])
                        if metric == "robust_acc":
                            n = minority_n
                        elif metric == "avg_acc":
                            n = total_n

                        stddev = np.sqrt(acc * (1 - acc) / n)
                        results[metric][split][epoch_save_str] = (acc, stddev)

                        if output:
                            print(
                                f"{metric_str} {split:<5} acc ({epoch_print_str} {epoch}): "
                                f"{acc*100:.1f} ({stddev*100:.1f})")
    return results


def process_df(train_df, val_df, test_df, n_groups):
    loss_metrics = []
    acc_metrics = []
    for group_idx in range(n_groups):  # 4 groups
        loss_metrics.append(f"avg_loss_group:{group_idx}")
        acc_metrics.append(f"avg_acc_group:{group_idx}")
    # robust acc
    for df in [train_df, val_df, test_df]:
        try:
            df["robust_loss"] = np.max(df.loc[:, loss_metrics], axis=1)
            df["robust_acc"] = np.min(df.loc[:, acc_metrics], axis=1)
        except:
            pass


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="a name for the experiment directory",
    )
    parser.add_argument("--dataset",
                        type=str,
                        default="CUB",
                        help="CUB, CelebA, or MultiNLI")
    # Default arguments (don't change)
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--exp_substring", type=str, default="")

    args = parser.parse_args()
    
    if args.exp_name is None:
        exp_dir = join(args.results_dir, args.dataset)
        experiments = "\n".join(os.listdir(exp_dir))
        assert False, f"Experiment name is required, here are the experiments:\n{experiments}"

    # Set folders
    metadata_dir = os.path.join(args.results_dir, args.dataset,
                                      args.exp_name)

    # Accuracies from downstream runs
    args.training_output_dir = metadata_dir
    runs = [
        folder for folder in os.listdir(args.training_output_dir)
        if args.exp_substring in folder
    ]

    # Print robust val accuracies from downstream runs
    for run in runs:
        try:
            sub_exp_name = run
            
            training_output_dir = os.path.join(args.training_output_dir,
                                            sub_exp_name, "model_outputs")
            train_path = os.path.join(training_output_dir, "train.csv")
            val_path = os.path.join(training_output_dir, "val.csv")
            test_path = os.path.join(training_output_dir, "test.csv")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            group_count = np.max(np.array([col.split(":")[1] for col in val_df.columns if "_group" in col]).astype(int)) + 1
            
            process_df(train_df, val_df, test_df, n_groups=group_count)

            dfs = {}
            dfs["train"] = train_df
            dfs["val"] = val_df
            dfs["test"] = test_df
            
            print(f"Downstream Accuracies for {sub_exp_name} with {group_count} groups.")
            with open(training_output_dir + "/val_accuracies.txt", "a") as text_file:
                print(f"Downstream Accuracies for {sub_exp_name}", file=text_file)
                
            # Print average and worst group accuracies for val
            print_accs(
                dfs,
                args.training_output_dir,
                params=None,
                epoch_to_eval=None,
                print_avg=True,
                print_groups=True,
                output=True,
                splits=["val", 'test'],
                early_stop=True,
            )
            print("\n")
            
        except:
            import sys
            if str(sys.exc_info()[0]) != "<class 'FileNotFoundError'>":
                print("\n")
                print(f"problem with {run}")
                print(sys.exc_info())
                pass