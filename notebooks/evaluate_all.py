from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import auc, recall_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
from argparse import ArgumentParser
import os

target_fpr = 0.2


white = "White"
asian = "Asian"
black = "Black"
male = "Male"
female = "Female"

labels = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def get_boostrap_ci_for_full_experiment(
    targets: np.ndarray,
    predictions: np.ndarray,
    race: np.ndarray,
    sex: np.ndarray,
    n_models: int = 2000,
    level: float = 0.95,
):
    """
    Get all CIs for FPR/TPR/Youden/AUC per subgroup for a global threshold with target fpr of 0.2
    """
    n_models, n_samples = targets.shape[0], targets.shape[1]
    all_fpr, all_tpr, all_roc_auc, all_youden, all_f1 = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    sample_target, sample_pred = targets, predictions
    sample_race, sample_sex = race, sex

    for model in range(n_models):
        fpr, tpr, thres = roc_curve(sample_target[model, :], sample_pred[model, :])

        all_roc_auc["all"].append(auc(fpr, tpr))

        # Computing global threshold
        # idx_target_fpr_threshold = np.argmin(np.abs(fpr - target_fpr))
        f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
        max_f1_idx = np.argmax(f1_scores)
        max_f1_threshold = thres[max_f1_idx]
        
        op = thres[max_f1_idx]
        all_fpr["all"].append(fpr[max_f1_idx])
        all_tpr["all"].append(tpr[max_f1_idx])
        all_youden["all"].append(
            (tpr[max_f1_idx] - fpr[max_f1_idx])
        )
        all_f1['all'].append(f1_scores[max_f1_idx])

        # Getting race subbroups results
        for r in [white, asian, black]:
            targets_r, preds_r = (
                sample_target[model, sample_race == r],
                sample_pred[model, sample_race == r],
            )
            all_roc_auc[r].append(roc_auc_score(targets_r, preds_r))
            all_fpr[r].append(1 - recall_score(targets_r, preds_r >= op, pos_label=0))
            all_tpr[r].append(recall_score(targets_r, preds_r >= op, pos_label=1))
            all_youden[r].append(all_tpr[r][-1] - all_fpr[r][-1])
            all_f1[r].append(2 * (all_tpr[r][-1] * (1 - all_fpr[r][-1])) / (all_tpr[r][-1] + (1 - all_fpr[r][-1])))

        # all_roc_auc["difference"] = max([all_roc_auc[r][-1] for r in [white, asian, black]]) - min([all_roc_auc[r][-1] for r in [white, asian, black]])
        # all_fpr["difference"] = max([all_fpr[r][-1] for r in [white, asian, black]]) - min([all_fpr[r][-1] for r in [white, asian, black]])
        # all_tpr["difference"] = max([all_tpr[r][-1] for r in [white, asian, black]]) - min([all_tpr[r][-1] for r in [white, asian, black]])
        # all_youden["difference"] = max([all_youden[r][-1] for r in [white, asian, black]]) - min([all_youden[r][-1] for r in [white, asian, black]])
        # all_f1["difference"] = max([all_f1[r][-1] for r in [white, asian, black]]) - min([all_f1[r][-1] for r in [white, asian, black]])

        # # Getting sex subgroup results
        # for s in [male, female]:
        #     targets_s, preds_s = (
        #         sample_target[model, sample_sex == s],
        #         sample_pred[model, sample_sex == s],
        #     )
        #     all_roc_auc[s].append(roc_auc_score(targets_s, preds_s))
        #     all_fpr[s].append(1 - recall_score(targets_s, preds_s >= op, pos_label=0))
        #     all_tpr[s].append(recall_score(targets_s, preds_s >= op, pos_label=1))
        #     all_youden[s].append(all_tpr[s][-1] - all_fpr[s][-1])
        #     all_f1[s].append(2 * (all_tpr[s][-1] * (1 - all_fpr[s][-1])) / (all_tpr[s][-1] + (1 - all_fpr[s][-1])))

    

    return {
        "AUC": {
            asian: np.mean(all_roc_auc[asian]),
            black: np.mean(all_roc_auc[black]),
            white: np.mean(all_roc_auc[white]),
            # male: np.mean(all_roc_auc[male]),
            # female: np.mean(all_roc_auc[female]),
            "difference": 
            max(np.mean(all_roc_auc[asian]), np.mean(all_roc_auc[black]), np.mean(all_roc_auc[white])) - 
            min(np.mean(all_roc_auc[asian]), np.mean(all_roc_auc[black]), np.mean(all_roc_auc[white])),
            "all": np.mean(all_roc_auc["all"]),
        },
        "F1": {
            asian: np.mean(all_f1[asian]),
            black: np.mean(all_f1[black]),
            white: np.mean(all_f1[white]),
            # male: np.mean(all_f1[male]),
            # female: np.mean(all_f1[female]),
            "difference":
            max(np.mean(all_f1[asian]), np.mean(all_f1[black]), np.mean(all_f1[white])) -
            min(np.mean(all_f1[asian]), np.mean(all_f1[black]), np.mean(all_f1[white])),
            "all": np.mean(all_f1["all"]),
        },
        "TPR": {
            white: np.mean(all_tpr[white]),
            black: np.mean(all_tpr[black]),
            asian: np.mean(all_tpr[asian]),
            # male: np.mean(all_tpr[male]),
            # female: np.mean(all_tpr[female]),
            "difference":
            max(np.mean(all_tpr[asian]), np.mean(all_tpr[black]), np.mean(all_tpr[white])) -
            min(np.mean(all_tpr[asian]), np.mean(all_tpr[black]), np.mean(all_tpr[white])),
            "all": np.mean(all_tpr["all"]),
        },
        "FPR": {
            black: np.mean(all_fpr[black]),
            white: np.mean(all_fpr[white]),
            asian: np.mean(all_fpr[asian]),
            # male: np.mean(all_fpr[male]),
            # female: np.mean(all_fpr[female]),
            "difference":
            max(np.mean(all_fpr[asian]), np.mean(all_fpr[black]), np.mean(all_fpr[white])) -
            min(np.mean(all_fpr[asian]), np.mean(all_fpr[black]), np.mean(all_fpr[white])),
            "all": np.mean(all_fpr["all"]),
        },
        # "Youden's Index": {
        #     black: np.mean(all_youden[black]),
        #     white: np.mean(all_youden[white]),
        #     asian: np.mean(all_youden[asian]),
        #     male: np.mean(all_youden[male]),
        #     female: np.mean(all_youden[female]),
        #     "all": np.mean(all_youden["all"]),
        # },
    }

def get_df(cnn_preds, data_characteristics, label):
    # PARAMETERS FOR CI
    n_models = len(cnn_preds)

    preds = []
    targets = []
    for df in cnn_preds:
        preds.append(df["class_" + str(label)])
        targets.append(df["target_" + str(label)])

    preds = np.array(preds)
    targets = np.array(targets)
    race = data_characteristics.race.values
    sex = data_characteristics.sex.values

    results = get_boostrap_ci_for_full_experiment(
        targets=targets,
        predictions=preds,
        race=race,
        sex=sex,
        n_models=n_models,
    )

    # columns_as_in_manuscript = [white, asian, black, female, male, "all"]
    columns_as_in_manuscript = [white, asian, black, "difference", "all"]
    res_df = pd.DataFrame.from_dict(results, orient="index")[
        columns_as_in_manuscript
    ]
    return res_df


def print_results(res_df, label, ci_level):
    print(
            f"\nResults for: {labels[label].upper()} ({ci_level * 100:.0f}%-CI)"
        )
    rounded_df = res_df.map(lambda x: f"{x * 100:.1f}%")
    print(tabulate(rounded_df, headers=rounded_df.columns))


def compute_results(extractor, mode, sampler='none'):
    cnn_preds = []
    folder_path = f"../prediction/{extractor}_{mode}"
    if sampler != 'none':
        folder_path += f"_{sampler}"
    for file_name in os.listdir(folder_path):
        if file_name.endswith("predictions.test.csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            cnn_preds.append(df)

    ci_level = 0.95
    n_models = len(cnn_preds)
    label_list = [0,10,2,9]

    data_characteristics = pd.read_csv("../datafiles/chexpert/chexpert.resample.test.csv")

    # GET RESULTS
    for label in label_list:
        res_df = get_df(cnn_preds, data_characteristics, label)
        print_results(res_df, label, ci_level)


def compare_results(extractor, mode):
    label_list = [0,10,2,9]
    ci_level = 0.95

    cnn_preds_1 = []
    folder_path = f"../prediction/{extractor}_baseline"
    for file_name in os.listdir(folder_path):
        if file_name.endswith("predictions.test.csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            cnn_preds_1.append(df)

    cnn_preds_2 = []
    folder_path = f"../prediction/{extractor}_{mode}"
    for file_name in os.listdir(folder_path):
        if file_name.endswith("predictions.test.csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            cnn_preds_2.append(df)

    n_models_1 = len(cnn_preds_1)
    n_models_2 = len(cnn_preds_2)

    data_characteristics = pd.read_csv("../datafiles/chexpert/chexpert.resample.test.csv")

    # GET RESULTS
    for label in label_list:
        res_df_1 = get_df(cnn_preds_1, data_characteristics, label)
        res_df_2 = get_df(cnn_preds_2, data_characteristics, label)
        relative = res_df_2 - res_df_1
        print_results(relative, label, ci_level)


def main(args):
    # print("Averaged results for baseline model: {}".format(args.extractor))
    # compute_results(args.extractor, 'baseline')
    # if not args.mode == 'baseline':
    #     print("Averaged results for {} model: {}".format(args.mode, args.extractor))
    #     compute_results(args.extractor, args.mode)
    #     print("Averaged comparison between baseline and {}: {}".format(args.mode, args.extractor))
    #     compare_results(args.extractor, args.mode)$
    print("Averaged results for baseline model: {}".format(args.extractor))
    compute_results(args.extractor, 'baseline')

    print("\n \n")

    print("Averaged results for {} model: {} - sampler: {}".format(args.mode, args.extractor, args.sampler))
    compute_results(args.extractor, args.mode, sampler=args.sampler)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--extractor', default='densenet')
    parser.add_argument('--mode', default='baseline')
    parser.add_argument('--sampler', default='none')
    args = parser.parse_args()

    main(args)