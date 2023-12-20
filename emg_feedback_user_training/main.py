import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from emg_feedback_user_training.data import ExptCondition, FreeResults, ProcessedSubjectData, Subject, load_dataset
from emg_feedback_user_training.similarities import (
    compute_feature_similarities,
    median_heuristic_gamma,
    normalize_sims,
    scalar_similarity_measure,
)
from emg_feedback_user_training.utils import GESTURE_NAMES, PROJECT_PATH, savefig

DECISION_THRESHOLD = 0.5
GESTURES_AND_NOCLASS = GESTURE_NAMES + ["NoClass"]

# # https://community.plotly.com/t/plotly-colours-list/11730/2
COLORS = {
    ExptCondition.Control: "#EF8677",  # pastel red
    ExptCondition.Veridical: "#A0E77D",  # pastel green
    ExptCondition.Modified: "#82B6D9",  # pastel blue
}


def make_heatmap(
    data: np.ndarray,
    display_text: np.ndarray = None,
    title=None,
    xlabels=GESTURE_NAMES,
    ylabels=GESTURE_NAMES,
    xaxis_title=None,
    yaxis_title=None,
    zmin=None,
    zmax=None,
    tril=False,
    exclude_zeros=False,
    colorscale="Viridis",
) -> go.Figure:
    if data.shape[0] != len(ylabels) or data.shape[1] != len(xlabels):
        raise ValueError(f"Mismatched shapes: {data.shape=}, {len(xlabels)=}, {len(ylabels)=}")

    # By default, show numerical values with 2 decimal places
    text = data
    texttemplate = "%{text:.2f}"
    if display_text is not None:
        # If requested, show text exactly as given
        if display_text.shape != data.shape:
            raise ValueError(f"Mismatched shapes: {display_text.shape=}, {data.shape=}")
        if display_text.dtype != object:
            raise ValueError("display_text must be of type object")
        text = display_text
        texttemplate = "%{text}"

    if tril:
        data[np.triu_indices(data.shape[0], k=1)] = None
    if exclude_zeros:
        data[data == 0] = None

    fig = px.imshow(data, y=ylabels, x=xlabels, color_continuous_scale=colorscale, aspect="equal", zmin=zmin, zmax=zmax)

    if title is not None:
        fig.update_layout(title=title)

    fig.update_traces(text=text, texttemplate=texttemplate)
    fig.update_layout(xaxis_title=xaxis_title)
    fig.update_layout(yaxis_title=yaxis_title)
    # fig.update_xaxes(side="top")
    return fig


def compute_accuracies(one_hot_labels, probs, game_lens) -> Tuple[List[float], float]:
    """Get acc for each game, and total acc.
    NOTE - correct moves must be above threshold and correct class."""
    # Result for each game
    game_accs = []
    cursor = 0
    for game_len in game_lens:
        labels = one_hot_labels[cursor : cursor + game_len].argmax(-1)
        game_probs = probs[cursor : cursor + game_len]
        preds = game_probs.argmax(-1)

        is_above_thresh = game_probs.max(-1) > DECISION_THRESHOLD
        correct = np.sum(preds[is_above_thresh] == labels[is_above_thresh])
        game_accs.append(correct / game_len)
        cursor += game_len

    # Result across pooled games
    all_preds = probs.argmax(-1)
    all_labels = one_hot_labels.argmax(-1)
    is_above_thresh = probs.max(-1) > DECISION_THRESHOLD  # Indices where decision was made
    correct = np.sum(all_preds[is_above_thresh] == all_labels[is_above_thresh])
    total = len(all_labels)
    total_acc = correct / total

    return np.array(game_accs), total_acc


def _compute_one_conf_mat(one_hot_labels, probs):
    classes = GESTURE_NAMES + ["NoClass"]
    integer_labels = one_hot_labels.argmax(-1)

    # Compute preds, assigning "below threshold" to a new "NoClass" class
    below_thresh_idx = probs.max(-1) < DECISION_THRESHOLD
    preds = np.empty(len(one_hot_labels), dtype=int)
    preds[below_thresh_idx] = len(classes) - 1  # Assign to the "NoClass" class
    preds[~below_thresh_idx] = probs[~below_thresh_idx].argmax(-1)
    return confusion_matrix(integer_labels, preds, labels=np.arange(len(classes)))


def compute_conf_mats(one_hot_labels, probs, game_lens):
    first3_len = sum(game_lens[:3])
    first3_result = _compute_one_conf_mat(one_hot_labels[:first3_len], probs[:first3_len])
    all_result = _compute_one_conf_mat(one_hot_labels, probs)
    return first3_result, all_result


def compute_clf_report(one_hot_labels, probs):
    classes = GESTURE_NAMES + ["NoClass"]
    below_thresh_idx = probs.max(-1) < DECISION_THRESHOLD
    preds = np.empty(len(one_hot_labels), dtype=int)
    preds[below_thresh_idx] = len(classes) - 1
    preds[~below_thresh_idx] = probs[~below_thresh_idx].argmax(-1)

    integer_labels = one_hot_labels.argmax(-1)

    return classification_report(
        integer_labels,
        preds,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )


def compute_baseline_acc_and_conf_mat(subj_data: ProcessedSubjectData) -> float:
    """Baseline accuracy, using model trained on calibration and evaluated on instructed games"""
    calib_model = subj_data.calibration_model
    instr_data = subj_data.instructed.features
    instr_labels_one_hot = subj_data.instructed.one_hot_labels
    instr_labels = instr_labels_one_hot.argmax(-1)

    probs = calib_model.predict_proba(instr_data)
    preds = probs.argmax(-1)
    is_above_thresh = probs.max(-1) > DECISION_THRESHOLD
    correct = np.sum(preds[is_above_thresh] == instr_labels[is_above_thresh])
    acc = correct / len(instr_labels)

    # NOTE - this confusion matrix gives pretty noisy estimates,
    # since we have only 3 examples from each class during instructed games
    baseline_conf_mat = _compute_one_conf_mat(instr_labels_one_hot, probs)
    return acc, baseline_conf_mat


def compute_baseline_sims(subj_data: ProcessedSubjectData, method: str, gamma: float) -> np.ndarray:
    """Baseline similarities, using all data from calibration and instructed blocks"""
    features = np.concatenate([subj_data.calibration.features, subj_data.instructed.features])
    one_hot_labels = np.concatenate([subj_data.calibration.one_hot_labels, subj_data.instructed.one_hot_labels])
    sims = compute_feature_similarities(features, one_hot_labels, method=method, gamma=gamma)
    return sims


@logger.catch(onerror=lambda _: sys.exit(1))
def analyze(
    subj: Subject,
    subj_data: ProcessedSubjectData,
    similarity_method: str,
    within_class_numerator: bool,
    normalized: bool,
):
    """Process one subject"""
    # For each block, compute feature space similarities
    rbf_median_gamma = median_heuristic_gamma(
        np.concatenate([x.features for x in [subj_data.calibration, subj_data.instructed, subj_data.free]])
    )

    # Compute metrics for each game of the "Free" block and for entire pooled "Free" block
    free = subj_data.free
    game_accs, overall_acc = compute_accuracies(free.one_hot_labels, free.probs, free.game_lens)
    overall_sims = compute_feature_similarities(
        features=free.features, one_hot_labels=free.one_hot_labels, method=similarity_method, gamma=rbf_median_gamma
    )

    free_first3_conf_mat, free_overall_conf_mat = compute_conf_mats(free.one_hot_labels, free.probs, free.game_lens)

    clf_report = compute_clf_report(free.one_hot_labels, free.probs)
    baseline_acc, baseline_conf_mat = compute_baseline_acc_and_conf_mat(subj_data)
    baseline_sims = compute_baseline_sims(subj_data, similarity_method, gamma=rbf_median_gamma)
    overall_sims, baseline_sims = normalize_sims([overall_sims, baseline_sims], normalized=normalized)

    overall_sim_scalar = scalar_similarity_measure(overall_sims, within_class_numerator)
    baseline_sim_scalar = scalar_similarity_measure(baseline_sims, within_class_numerator)

    return FreeResults(
        subj=subj,
        game_accs=game_accs,
        overall_acc=overall_acc,
        baseline_conf_mat=baseline_conf_mat,
        free_first3_conf_mat=free_first3_conf_mat,
        free_overall_conf_mat=free_overall_conf_mat,
        sim_method=similarity_method,
        overall_sims=overall_sims,
        overall_sim_scalar=overall_sim_scalar,
        clf_report=clf_report,
        baseline_acc=baseline_acc,
        baseline_sims=baseline_sims,
        baseline_sim_scalar=baseline_sim_scalar,
    )


@logger.catch(onerror=lambda _: sys.exit(1))
def make_plots_overall(all_results: Dict[ExptCondition, List[FreeResults]], output_path: Path):
    for condition in ExptCondition:
        subset = all_results[condition]
        if len(subset) == 0:
            logger.warning(f"No results for: {condition}, skipping...")
            continue

        # Average overall Confusion Matrix
        avg_overall_conf_mat = np.mean(np.stack([x.free_overall_conf_mat for x in subset]), axis=0)
        avg_overall_conf_mat = avg_overall_conf_mat[:-2]  # Remove "Rest" and "NoClass" rows
        avg_overall_conf_mat /= avg_overall_conf_mat.sum(axis=1, keepdims=True)  # normalize rows
        fig = make_heatmap(
            avg_overall_conf_mat,
            title=condition.name,
            xlabels=GESTURES_AND_NOCLASS,
            ylabels=GESTURES_AND_NOCLASS[:-2],
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            zmin=0,
            zmax=1,
            colorscale="Blues",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        savefig(fig, output_path, f"overall_conf_mat.{condition.name}")

        similarity_method = subset[0].sim_method

        # Average similarity matrix (ignoring "Rest")
        avg_overall_sims = np.nanmean([x.overall_sims[:-1, :-1] for x in subset], axis=0)
        fig = make_heatmap(
            avg_overall_sims,
            title=condition.name,
            xlabels=GESTURE_NAMES[:-1],
            ylabels=GESTURE_NAMES[:-1],
            zmin=0.0,
            zmax=1.0,
            tril=True,
            colorscale="Blues",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        savefig(fig, output_path, f"overall_similarities.{similarity_method}.{condition.name}")

        # Average baseline-subtracted similarity matrix
        avg_overall_sims = np.nanmean([x.overall_sims[:-1, :-1] - x.baseline_sims[:-1, :-1] for x in subset], axis=0)
        # NOTE - global zmin, zmax should be adjusted after collecting new data for better contrast/easier viewing
        fig = make_heatmap(
            avg_overall_sims,
            title=condition.name,
            xlabels=GESTURE_NAMES[:-1],
            ylabels=GESTURE_NAMES[:-1],
            zmin=-0.25,
            zmax=0.25,
            tril=True,
            colorscale="Blues",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        savefig(fig, output_path, f"overall_similarities.baseline_subtracted.{similarity_method}.{condition.name}")

    # Group-level baseline acc and baseline sims
    rows = []
    for condition in ExptCondition:
        subset = all_results[condition]
        for item in subset:
            row = {}
            row["condition"] = condition
            row["subj_id"] = item.subj.subject_id
            row["baseline_acc"] = item.baseline_acc
            row["baseline_sim_scalar"] = item.baseline_sim_scalar
            row["overall_acc"] = item.overall_acc
            row["overall_sim_scalar"] = item.overall_sim_scalar

            rows.append(row)
    df = pd.DataFrame(rows)

    # Baseline values of accuracy and class separation
    fig = make_subplots(rows=1, cols=2)
    box_kw = dict(boxpoints="all", showlegend=False, boxmean=True)
    for condition, color in COLORS.items():
        subset = df[df["condition"] == condition]
        y = subset["baseline_acc"]
        x = [condition.name] * len(y)
        fig.add_trace(go.Box(x=x, y=y, marker_color=color, **box_kw), row=1, col=1)

        y = subset["baseline_sim_scalar"]
        x = [condition.name] * len(y)
        fig.add_trace(go.Box(x=x, y=y, marker_color=color, **box_kw), row=1, col=2)
    fig.update_layout(
        title="Group-level Baselines",
        yaxis=dict(range=(0, 1), title=r"$\textrm{Accuracy}$", title_standoff=0),
        yaxis2=dict(title=r"$d_{sep}$", title_standoff=0),
        margin=dict(l=5, r=0, b=0, t=50),
    )
    savefig(fig, output_path, f"group-baselines-bar.{similarity_method}")

    # Change from baseline values of accuracy and class separation
    fig = make_subplots(rows=1, cols=2)
    for condition, color in COLORS.items():
        subset = df[df["condition"] == condition]
        y = subset["overall_acc"] - subset["baseline_acc"]
        x = [condition.name] * len(y)
        fig.add_trace(go.Box(x=x, y=y, marker_color=color, **box_kw), row=1, col=1)
        if condition == ExptCondition.Modified:
            fig.add_annotation(text="*", x=condition.name, y=max(y) + 0.03, showarrow=False, row=1, col=1)

        y = subset["overall_sim_scalar"] - subset["baseline_sim_scalar"]
        x = [condition.name] * len(y)
        fig.add_trace(go.Box(x=x, y=y, marker_color=color, **box_kw), row=1, col=2)
    fig.update_layout(
        title="Additive Changes from Baseline",
        yaxis=dict(title=r"$\Delta \textrm{Accuracy}$", title_standoff=0),
        yaxis2=dict(title=r"$\Delta d_{sep}$", title_standoff=0),
        margin=dict(l=5, r=0, b=0, t=50),
    )
    savefig(fig, output_path, f"group-overall-change-bar.{similarity_method}")


def make_report_each_subj(all_results: Dict[ExptCondition, List[FreeResults]], output_path: Path):
    """Export performance metrics in convenient format for statistical comparisons"""
    df = []
    for condition in ExptCondition:
        try:
            subset = all_results[condition]
        except KeyError:
            logger.warning(f"No results for: {condition}, skipping...")
            continue
        similarity_method = subset[0].sim_method
        for result in subset:
            row = {}
            row["subject_id"] = result.subj.subject_id
            row["condition"] = condition.name
            row["acc"] = result.overall_acc
            row["acc_baseline"] = result.baseline_acc
            row["acc_change"] = result.overall_acc - result.baseline_acc
            row["acc_change_percent"] = (result.overall_acc - result.baseline_acc) / result.baseline_acc
            row["sim"] = result.overall_sim_scalar
            row["sim_baseline"] = result.baseline_sim_scalar
            row["sim_change"] = result.overall_sim_scalar - result.baseline_sim_scalar
            row["sim_change_percent"] = (
                result.overall_sim_scalar - result.baseline_sim_scalar
            ) / result.baseline_sim_scalar
            df.append(row)
    df = pd.DataFrame(df)
    df.to_csv(output_path / f"each_subj_results.{similarity_method}.csv")


def run_once(*, names, clusters, one_hot_labels, title, outdir, xaxis, yaxis, zaxis):
    n_class = len(clusters)
    fig = go.Figure()

    for name, cluster in zip(names, clusters):
        fig.add_trace(
            go.Scatter3d(
                x=cluster[:, 0],
                y=cluster[:, 1],
                z=cluster[:, 2],
                name=name,
                mode="markers",
                marker_size=15,
                marker_opacity=0.8,
            ),
        )

    all_data = np.concatenate(clusters)
    gamma = median_heuristic_gamma(all_data)
    sims = compute_feature_similarities(all_data, one_hot_labels, method="rbf", gamma=gamma, n_class=n_class)
    sims = normalize_sims([sims], normalized=True)[0]
    scalar_sim = scalar_similarity_measure(sims, True)
    fig.update_layout(
        scene=dict(
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=zaxis,
            xaxis_title="RMS Channel 1",
            yaxis_title="RMS Channel 4",
            zaxis_title="RMS Channel 7",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.15)),
            annotations=[
                dict(
                    showarrow=False,
                    x=0.05,
                    y=0.05,
                    z=0,
                    text=f"d<sub>sep</sub>={round(scalar_sim, 2)}",
                    font=dict(size=25),
                ),
            ],
        ),
        autosize=False,
        width=600,
        height=600,
        margin=dict(l=10, r=10, b=10, t=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="center", x=0.98),
        font_size=20,
    )

    fig2 = make_heatmap(sims, xlabels=names, ylabels=names, zmin=0, zmax=1, tril=True, colorscale="Blues")
    fig2.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    savefig(fig, outdir, f"toy-similarities-scatter-{title}", png=True, html=False, pdf=False)
    savefig(fig2, outdir, f"toy-similarities-heatmap-{title}", png=True, html=False, pdf=False)


def get_subj(subj_data_dict, subj_id):
    for s in subj_data_dict.keys():
        if s.subject_id.lower() == subj_id:
            return subj_data_dict[s]
    raise ValueError(f"subj_id {subj_id} not found")


def example_similarities():
    results_path = PROJECT_PATH / "results" / "figures"
    results_path.mkdir(exist_ok=True, parents=True)

    subj_data_dict = load_dataset()

    chosen = get_subj(subj_data_dict, "subj_044")
    subset_idx = np.array([0, 3, 6])
    features_before = np.concatenate([chosen.calibration.features, chosen.instructed.features])[:, subset_idx]
    labels_before = np.concatenate([chosen.calibration.one_hot_labels, chosen.instructed.one_hot_labels])
    features_after = chosen.free.features[:, subset_idx]
    labels_after = chosen.free.one_hot_labels

    # Same axes for both plots
    xaxis = dict(range=[0, 0.1], tickmode="array", tickvals=[0, 0.05, 0.1], tickfont=dict(size=15))
    yaxis = dict(range=[0, 0.1], tickmode="array", tickvals=[0, 0.05, 0.1], tickfont=dict(size=15))
    zaxis = dict(range=[0, 0.1], tickmode="array", tickvals=[0, 0.05, 0.1], tickfont=dict(size=15))
    names = ["Left", "Down", "Right"]  # Example gestures

    # Make plots of "before"
    clusters = []
    one_hot_labels = []
    for i, n in enumerate(names):
        idx = np.where(labels_before.argmax(-1) == GESTURE_NAMES.index(n))[0]
        clusters.append(features_before[idx])
        one_hot_tmp = np.zeros((len(idx), len(names)))
        one_hot_tmp[:, i] = 1
        one_hot_labels.append(one_hot_tmp)
    one_hot_labels = np.concatenate(one_hot_labels)
    run_once(
        names=names,
        clusters=clusters,
        one_hot_labels=one_hot_labels,
        title="before",
        outdir=results_path,
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
    )

    # Make plots of "after"
    clusters = []
    one_hot_labels = []
    for i, n in enumerate(names):
        idx = np.where(labels_after.argmax(-1) == GESTURE_NAMES.index(n))[0]
        clusters.append(features_after[idx])
        one_hot_tmp = np.zeros((len(idx), len(names)))
        one_hot_tmp[:, i] = 1
        one_hot_labels.append(one_hot_tmp)
    one_hot_labels = np.concatenate(one_hot_labels)
    run_once(
        names=names,
        clusters=clusters,
        one_hot_labels=one_hot_labels,
        title="after",
        outdir=results_path,
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
    )


def rpm(y, k=0.75):
    res = np.copy(y)
    res **= k
    res /= res.sum()
    return res


def example_modified_probs():
    results_path = PROJECT_PATH / "results" / "figures"
    results_path.mkdir(exist_ok=True, parents=True)

    fig = go.Figure()
    y1 = np.array([0, 0.02, 0.04, 0, 0, 0.14, 0.75, 0, 0.05])
    y2 = rpm(y1)
    fig.add_trace(go.Bar(x=GESTURE_NAMES, y=y1, name="Veridical", marker_color="#A0E77D"))
    fig.add_trace(go.Bar(x=GESTURE_NAMES, y=y2, name="Modified", marker_color="#82B6D9"))
    fig.add_hline(y=0.5)
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=np.linspace(0, 1, 11), range=[0, 1]),
        margin=dict(l=0, r=10, b=0, t=10),
    )
    savefig(fig, results_path, "veridical_and_modified", height=450, png=True, html=False, pdf=False)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    results_path = PROJECT_PATH / "results" / "figures"
    results_path.mkdir(exist_ok=True, parents=True)

    subj_data_dict = load_dataset()

    # For consistent interpretation, compute a scalar similarity value where larger values are "better"
    # - For "avg_pairwise_l2", smaller within_class is good. Put on denominator so that larger ratio values are "better"
    # - For "rbf", larger within_class is good. Put on numerator so larger ratio values are "better"
    methods = [
        ("rbf", True, True),
        # ("mahalanobis", False, True),
        # ("ncut", True, True),
        # ("cos", True, True),
        # ("avg_pairwise_l2", False, True),
    ]
    for similarity_method, within_class_num, normalized in tqdm(methods, desc="similarity Method", leave=True):
        all_results = {condition: [] for condition in ExptCondition}

        for subj, subj_data in tqdm(subj_data_dict.items(), "analyze each subject", position=1, leave=False):
            result = analyze(subj, subj_data, similarity_method, within_class_num, normalized=normalized)
            all_results[subj.experiment_condition].append(result)

        make_plots_overall(all_results, results_path)
        make_report_each_subj(all_results, results_path)

    example_similarities()
    example_modified_probs()


if __name__ == "__main__":
    main()
