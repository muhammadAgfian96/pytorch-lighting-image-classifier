import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from clearml import Task
from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_tpr_fpr(y_real, y_pred):
    """
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    """

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr


def get_all_roc_coordinates(y_real, y_proba):
    """
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    """
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, scatter=True, ax=None):
    """
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    """
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color="green", ax=ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


# Generate Graph
def generate_plot_one_vs_rest(
    class_names: list,
    gt_labels: list,
    preds_softmax: np.ndarray,
    path_to_save="logger_roc",
    task: Task | None = None,
    **kwargs,
):
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    print("Generating plots for One vs Rest")
    if task is None:
        return
    os.makedirs(path_to_save, exist_ok=True)
    ls_plts = []
    ls_path_figures = []

    n_classes = len(class_names)
    n_cols = 6 if n_classes >= 6 else n_classes
    n_rows = int(np.ceil(n_classes / n_cols)) * 2

    fig_width = n_cols * 4
    fig_height = n_rows * 4
    plt.figure(figsize=(fig_width, fig_height))

    bins = [i / 25 for i in range(25)] + [1]
    roc_auc_ovr = {}
    fig_count = 0
    i = 0

    start_pos_top = 0
    start_pos_bottom = n_cols
    for idx in range(len(class_names)):
        pos_top = start_pos_top + i + 1
        pos_bottom = start_pos_bottom + i + 1

        if (i + 1) % n_cols == 0:
            i = -1
            start_pos_top += n_cols * 2
            start_pos_bottom += n_cols * 2

        # Gets the class
        c = class_names[idx]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux["class"] = [1 if y == c else 0 for y in gt_labels]
        df_aux["prob"] = preds_softmax[:, idx]
        df_aux = df_aux.reset_index(drop=True)
        # print(df_aux.head())

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(n_rows, n_cols, pos_top)
        sns.histplot(x="prob", data=df_aux, hue="class", color="b", ax=ax, bins=bins)
        ax.set_title(f"{c}", fontweight="bold", fontsize=14)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(n_rows, n_cols, pos_bottom)
        tpr, fpr = get_all_roc_coordinates(df_aux["class"], df_aux["prob"])
        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)

        # # Calculates the ROC AUC OvR
        try:
            roc_auc_ovr[c] = roc_auc_score(df_aux["class"], df_aux["prob"])
            ax_bottom.set_title(
                f"ROC Curve OvR: (AUC = {roc_auc_ovr[c]:.3f})", fontsize=10
            )
        except ValueError as e:
            print(f"Error calculating the ROC AUC OvR for class {c}: {e}")
            print(df_aux.head())
            ax_bottom.set_title("ROC Curve OvR: (AUC = 'Error')", fontsize=10)
        except Exception as e:
            print(f"Error calculating the ROC AUC OvR for class {c}: {e}")
            print(df_aux.head())
            ax_bottom.set_title("ROC Curve OvR: (AUC = 'Error')", fontsize=10)
        i += 1

    plt.tight_layout()
    filepath_fig = os.path.join(path_to_save, f"roc_ovr_{fig_count}.png")
    plt.savefig(filepath_fig)
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            title=kwargs.get("title", "ROC OvR"),
            series=kwargs.get("series", "ROC"),
            iteration=kwargs.get("iteration", 0),
            figure=plt,
        )
    ls_path_figures.append(filepath_fig)
    ls_plts.append(plt.gcf())

    return


def generate_plot_one_vs_one(
    class_names,
    gt_labels,
    preds_softmax,
    path_to_save="logger_roc_ovo",
    task: Task | None = None,
    **d_task,
):
    # Plots the Probability Distributions and the ROC Curves One vs One
    if task is None:
        return
    os.makedirs(path_to_save, exist_ok=True)
    print("Generating Plots One vs One")

    # Generates combinations of classes
    classes_combinations = []
    class_list = list(class_names)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            classes_combinations.append([class_list[i], class_list[j]])
            classes_combinations.append([class_list[j], class_list[i]])

    # Plots the Probability Distributions and the ROC Curves One vs ONe
    n_classes = len(class_names)
    n_combination = len(classes_combinations)

    n_cols = n_combination if n_combination <= 8 else 8
    n_rows = (
        int(np.ceil(n_combination / n_cols)) * 2
        if int(np.ceil(n_combination / n_cols)) <= 10
        else 10
    )

    print("n_rows, n_cols", n_rows, n_cols)
    # max_plot_in_fig = 8*10
    max_plot_in_fig = n_cols * n_rows

    fig_width = n_cols * 4
    fig_height = n_rows * 4
    figure = plt.figure(figsize=(fig_width, fig_height))

    bins = [i / 25 for i in range(25)] + [1]
    roc_auc_ovo = {}

    i = 0
    fig_count = 1
    ls_path_figures = []
    ls_plts = []
    start_pos_top = 0
    start_pos_bottom = n_cols
    count_graph = 0
    print(len(classes_combinations))
    for idx in range(len(classes_combinations)):
        pos_top = start_pos_top + i + 1
        pos_bottom = start_pos_bottom + i + 1

        if (i + 1) % n_cols == 0:
            i = -1
            start_pos_top += n_cols * 2
            start_pos_bottom += n_cols * 2

        if pos_bottom % max_plot_in_fig == 0:
            plt.tight_layout()
            filepath_fig = os.path.join(path_to_save, f"roc_ovo_{fig_count}.png")
            plt.savefig(filepath_fig)
            ls_path_figures.append(filepath_fig)
            ls_plts.append(figure)
            if task is not None:
                task.get_logger().report_matplotlib_figure(
                    title=d_task.get("title", "ROC OvO"),
                    series=d_task.get("series", "ROC OvO") + f"{fig_count}",
                    iteration=d_task.get("iteration", 0),
                    figure=plt,
                )
            plt.close()

            figure = plt.figure(figsize=(fig_width, fig_height))
            i = 0
            start_pos_top = 0
            start_pos_bottom = n_cols
            pos_top = start_pos_top + i + 1
            pos_bottom = start_pos_bottom + i + 1
            fig_count += 1

        # print('pos_top, pos_bottom', pos_top, pos_bottom)

        # Gets the class
        comb = classes_combinations[idx]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = class_list.index(c1)
        title = f"{c1} vs {c2}"

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux["class"] = gt_labels
        df_aux["prob"] = np.array(preds_softmax)[:, c1_index]

        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux["class"] == c1) | (df_aux["class"] == c2)]
        df_aux["class"] = [1 if y == c1 else 0 for y in df_aux["class"]]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest

        # print(pos_top, pos_bottom)
        ax = plt.subplot(n_rows, n_cols, pos_top)
        sns.histplot(x="prob", data=df_aux, hue="class", color="b", ax=ax, bins=bins)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.legend([f"Class {c1}", f"Class {c2}"])
        ax.set_xlabel(f"P(x = {c1})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(n_rows, n_cols, pos_bottom)
        tpr, fpr = get_all_roc_coordinates(df_aux["class"], df_aux["prob"])
        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)

        # Calculates the ROC AUC OvO

        try:
            roc_auc_ovo[title] = roc_auc_score(df_aux["class"], df_aux["prob"])
            ax_bottom.set_title(f"ROC Curve OvO {roc_auc_ovo[title]:.3f}", fontsize=10)
        except ValueError as e:
            print(f"Error calculating the ROC AUC OvO: {e}")
            print(df_aux.head())
            ax_bottom.set_title("ROC Curve OvO: (AUC = 'Error')", fontsize=10)
        except Exception as e:
            print(f"Error calculating the ROC AUC OvO: {e}")
            print(df_aux.head())
            ax_bottom.set_title("ROC Curve OvO: (AUC = 'Error')", fontsize=10)

        i += 1
        count_graph += 1

    plt.tight_layout()
    filepath_fig = os.path.join(path_to_save, f"roc_ovo_{fig_count}.png")
    if task is not None:
        task.get_logger().report_matplotlib_figure(
            title=d_task.get("title", "ROC OvO"),
            series=d_task.get("series", "ROC OvO") + f"{fig_count}",
            iteration=d_task.get("iteration", 0),
            figure=plt,
        )
    plt.savefig(filepath_fig)
    ls_path_figures.append(filepath_fig)
    ls_plts.append(figure)
    plt.close()

    # Displays the ROC AUC for each class
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovo:
        avg_roc_auc += roc_auc_ovo[k]
        i += 1
        print(f"{k} ROC AUC OvO: {roc_auc_ovo[k]:.4f}")
    print(f"average ROC AUC OvO: {avg_roc_auc/i:.4f}")
    return
