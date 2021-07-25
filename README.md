# Just Train Twice: Improving Group Robustness without Training Group Information

This code implements the following paper: 

> [Just Train Twice: Improving Group Robustness without Training Group Information](https://arxiv.org/pdf/2107.09044.pdf)


## Environment

Create an environment with the following commands:
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Downloading Datasets

- **Waterbirds:** Download waterbirds from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) and put it in `jtt/cub`.
    - In that directory, our code expects `data/waterbird_complete95_forest2water2/` with `metadata.csv` inside.

- **CelebA:** Download CelebA from [here](https://www.kaggle.com/jessicali9530/celeba-dataset) and put it in `jtt/celebA`.
    - In that directory, our code expects the following files/folders:
        - data/list_eval_partition.csv
        - data/list_attr_celeba.csv
        - data/img_align_celeba/

- **MultiNLI:** Follow instructions [here](https://github.com/kohpangwei/group_DRO#multinli-with-annotated-negations) to download this dataset and put in `jtt/multinli`
    - In that directory, our code expects the following files/folders:
        - data/metadata_random.csv
        - glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli
        - glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli-mm
        - glue_data/MNLI/cached_train_bert-base-uncased_128_mnli

- **CivilComments:** This dataset can be downloaded from [here](https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/) and put it in `jtt/jigsaw`. In that directory, our code expects a folder `data` with the downloaded dataset.

## **Running our Method**

- Train the initial ERM model:
    - `python generate_downstream.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --method ERM`
        - Some useful optional args: `--n_epochs $EPOCHS --lr $LR --weight_decay $WD`. Other args, e.g. batch size, can be changed in generate_downstream.py.
        - Datasets: `CUB`, `CelebA`, `MultiNLI`, `jigsaw`
    - Bash execute the generated script for ERM inside `results/dataset/$EXPERIMENT_NAME`
- Once ERM is done training, run `python process_training.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --folder_name $ERM_FOLDER_NAME --lr $LR --weight_decay $WD --deploy`
- Bash execute the generated scripts that have `JTT` in their name.

## Monitoring Performance

- Run `python analysis.py --exp_name $PATH_TO_JTT_RUNS --dataset $DATASET`
    - The `$PATH_TO_JTT_RUNS` will look like `$EXPERIMENT_NAME+"/train_downstream_"+$ERM_FOLDER_NAME+"/final_epoch"+$FINAL_EPOCH`
- You can also track accuracies in train.csv, val.csv, and test.csv in the JTT directory or use wandb to monitor performance for all experiments (although this does not include subgroups of CivilComments-WILDS)

## Running ERM, Joint DRO, or Group DRO
- Run `python generate_downstream.py --exp_name $EXPERIMENT_NAME --dataset $DATASET --method $METHOD`
        - Some useful optional args: `--n_epochs $EPOCHS --lr $LR --weight_decay $WD`
        - Datasets: `CUB`, `CelebA`, `MultiNLI`, `jigsaw`
- Bash execute the generated script for the method inside `results/dataset/$EXPERIMENT_NAME`

## **Adding other datasets**

Add the following:

- A dataset file (similar to cub_dataset.py)
- Edit `process_training.py` to include the required args for your dataset and implement a way for getting the spurious features from the dataset.

## Sample Commands for running JTT on Waterbirds

```
python generate_downstream.py --exp_name CUB_sample_exp --dataset CUB --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --method ERM

bash results/CUB/CUB_sample_exp/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python process_training.py --exp_name CUB_sample_exp --dataset CUB --folder_name ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0 --lr 1e-05 --weight_decay 1.0 --final_epoch 60 --deploy

bash results/CUB/CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch50/JTT_upweight_100_epochs_300_lr_1e-05_weight_decay_1.0/job.sh

python analysis.py --exp_name CUB_sample_exp/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/ --dataset CUB
```

## Sample Commands for running JTT on CelebA

```
python generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 50 --lr 1e-5 --weight_decay 0.1 --method ERM

bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python process_training.py --exp_name CelebA_sample_exp --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1 --lr 1e-05 --weight_decay 0.1 --final_epoch 1 --deploy

sbatch results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/job.sh

python analysis.py --exp_name CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/ --dataset CelebA
```


## Sample Commands for running JTT on MultiNLI

```
python generate_downstream.py --exp_name MultiNLI_sample_exp --dataset MultiNLI --n_epochs 5 --lr 2e-5 --weight_decay 0 --method ERM

bash results/MultiNLI/MultiNLI_sample_exp/ERM_upweight_0_epochs_5_lr_2e-05_weight_decay_0/job.sh

python process_training.py --exp_name MultiNLI_sample_exp --dataset MultiNLI --folder_name ERM_upweight_0_epochs_5_lr_2e-05_weight_decay_0.0_nobert --lr 1e-05 --weight_decay 0.1 --final_epoch 2 --deploy

bash results/MultiNLI/MultiNLI_sample_exp/train_downstream_ERM_upweight_0_epochs_5_lr_2e-05_weight_decay_0.0/final_epoch2/JTT_upweight_4_epochs_5_lr_2e-05_weight_decay_0/job.sh

python analysis.py --exp_name MultiNLI_sample_exp/train_downstream_ERM_upweight_0_epochs_5_lr_2e-05_weight_decay_0.0/final_epoch2/ --dataset MultiNLI
```


## Sample Commands for running JTT on CivilComments-WILDS

```
python generate_downstream.py --exp_name jigsaw_sample_exp --dataset jigsaw --n_epochs 3 --lr 2e-5 --weight_decay 0 --method ERM --batch_size 24

bash results/jigsaw/jigsaw_sample_exp/ERM_upweight_0_epochs_3_lr_2e-05_weight_decay_0.0/job.sh

python process_training.py --exp_name jigsaw_sample_exp --dataset jigsaw --folder_name ERM_upweight_0_epochs_3_lr_2e-05_weight_decay_0.0 --lr 1e-05 --weight_decay 0.01 --final_epoch 2 --batch_size 16 --deploy

bash results/jigsaw/jigsaw_sample_exp/train_downstream_ERM_upweight_0_epochs_3_lr_2e-05_weight_decay_0.0/final_epoch2/JTT_upweight_6_epochs_3_lr_1e-05_weight_decay_0.01/job.sh

python analysis.py --exp_name jigsaw_sample_exp/train_downstream_ERM_upweight_0_epochs_3_lr_2e-05_weight_decay_0.0/final_epoch2/ --dataset jigsaw
```
