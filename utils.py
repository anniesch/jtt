import sys
import os
import torch
import numpy as np
import csv

import torch
import torch.nn as nn
import torchvision
from models import model_attributes


class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write("\n")


def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)


def get_model(model, pretrained, resume, n_classes, dataset, log_dir):
    if resume:
        model = torch.load(os.path.join(log_dir, "last_model.pth"))
        d = train_data.input_size()[0]
    elif model_attributes[model]["feature_type"] in (
            "precomputed",
            "raw_flattened",
    ):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            
            assert dataset == "MultiNLI"

            from pytorch_transformers import BertConfig, BertForSequenceClassification

            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained("bert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model
