import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import sys
import torch


def evaluate_a(predictions_test,gold_data):
    # evaluate performance on subtask a
    levels = ["HS"]
    ground_truth = gold_data

    predicted = predictions_test
    ground_truth["predicted"] = predicted

    # Check length files
    if (len(ground_truth) != len(predicted)):
        sys.exit('Prediction and gold data have different number of lines.')

    # Check predicted classes
    for c in levels:
        gt_class = list(ground_truth[c].value_counts().keys())
        for value in predicted:
            if not value in gt_class:
                sys.exit("Wrong value in " + c + " prediction column.")

    # Compute Performance Measures HS
    acc_hs = accuracy_score(ground_truth["HS"], ground_truth["predicted"])
    [p_nohs, p_hs], [r_nohs, r_hs], [f1_nohs, f1_hs], support = precision_recall_fscore_support(ground_truth["HS"], ground_truth["predicted"], pos_label = 1)
    p_macro, r_macro, f1_macro, support = precision_recall_fscore_support(ground_truth["HS"], ground_truth["predicted"], average = "macro")

    return acc_hs, p_hs, p_nohs, r_hs, r_nohs, f1_hs, f1_nohs, p_macro, r_macro, f1_macro


def evaluate_b(pred,gold):
    # evaluate performance on subtask b
    levels = ["HS", "TR", "AG"]

    ground_truth = gold
    predicted = pred

    # Check length files
    if (len(ground_truth) != len(predicted)):
        sys.exit('Prediction and gold data have different number of lines.')

    # Check predicted classes
    for c in levels:
        gt_class = list(ground_truth[c].value_counts().keys())
        if not (predicted[c].isin(gt_class).all()):
            sys.exit("Wrong value in " + c + " prediction column.")

    data = pd.merge(ground_truth, predicted, on="id")

    if (len(ground_truth) != len(data)):
        sys.exit('Invalid tweet IDs in prediction.')

    # Compute Performance Measures
    acc_levels = dict.fromkeys(levels)
    p_levels = dict.fromkeys(levels)
    r_levels = dict.fromkeys(levels)
    f1_levels = dict.fromkeys(levels)
    for l in levels:
        acc_levels[l] = accuracy_score(data[l + "_x"], data[l + "_y"])
        p_levels[l], r_levels[l], f1_levels[l], _ = precision_recall_fscore_support(data[l + "_x"], data[l + "_y"], average="macro")
    macro_f1 = np.mean(list(f1_levels.values()))

    # Compute Exact Match Ratio
    check_emr = np.ones(len(data), dtype=bool)
    for l in levels:
        check_label = data[l + "_x"] == data[l + "_y"]
        check_emr = check_emr & check_label
    emr = sum(check_emr) / len(data)

    return macro_f1, emr, acc_levels, p_levels, r_levels, f1_levels


def main():
    loaded_model = pickle.load(open(r"es_model.sav", "rb"))

    #get predicted labels for the test set
    pred_labels = loaded_model(test_ids, test_features_arr)
    predictions_test_binary = list()

    #round values: =<0.5 --> 0, > 0.5 --> 1
    for x in pred_labels:
        y = torch.round(x)
        y = y.item()
        predictions_test_binary.append(y)

    acc_hs, p_hs, p_nohs, r_hs, r_nohs, f1_hs, f1_nohs, p_macro, r_macro, f1_macro = evaluate_a(predictions_test_binary, dtf_test)
    print("\t".join(["{}".format(x) for x in ["acc.", "P (1)", "P (0)", "R (1)", "R (0)", "F1 (1)", "F1 (0)", "P (avg)", "R (avg)", "F1 (avg)"]]))
    print("\t".join(["{0:.3f}".format(x) for x in [acc_hs, p_hs, p_nohs, r_hs, r_nohs, f1_hs, f1_nohs, p_macro, r_macro, f1_macro]]))

    macro_f1, emr, acc_levels, p_levels, r_levels, f1_levels = evaluate_b(pred_dtf, dtf_test)
    print("\t".join(["{}".format(x) for x in ["acc_HS", "acc_TR", "acc_AG", "p_HS", "p_TR", "p_AG", "r_HS", "r_TR", "r_AG", "f1_HS", "f1_TR", "f1_AG", "emr", "macro_f1"]]))
    print("\t".join(["{0:.3f}".format(x) for x in [acc_levels["HS"], acc_levels["TR"], acc_levels["AG"], p_levels["HS"], p_levels["TR"], p_levels["AG"], r_levels["HS"], r_levels["TR"], r_levels["AG"], f1_levels["HS"], f1_levels["TR"], f1_levels["AG"], emr, macro_f1]]))


if __name__ == '__main__':
    main()