from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from emg_feedback_user_training.models.svmnet import SVMNet
from emg_feedback_user_training.utils import PROJECT_PATH


class ExptCondition(Enum):
    Control = 0
    Veridical = 1
    Modified = 2


@dataclass
class BlockContents:
    features: np.ndarray
    one_hot_labels: np.ndarray
    probs: Optional[np.ndarray]  # Only present for "Free" block
    game_lens: Optional[List[int]] = None  # "Calibration" does not have games


@dataclass(eq=True, frozen=True)
class Subject:
    subject_id: str
    experiment_condition: ExptCondition


@dataclass
class ProcessedSubjectData:
    """After determining offsets, load features, probs, and labels for each block"""

    calibration: BlockContents
    calibration_model: SVMNet
    instructed: BlockContents
    instructed_model: SVMNet
    free: BlockContents


@dataclass
class FreeResults:
    """Results for the "Free" block"""

    subj: Subject
    game_accs: List[float]
    overall_acc: float
    baseline_conf_mat: np.ndarray
    free_first3_conf_mat: List[np.ndarray]
    free_overall_conf_mat: np.ndarray
    sim_method: str
    overall_sims: np.ndarray  # Similarities between gesture classes during "Free"
    overall_sim_scalar: float
    clf_report: Dict[str, Any]
    # Using model trained on calibration, get accuracy on Instructed block
    baseline_acc: float
    baseline_sims: np.ndarray  # Similarities between gesture classes during calibration + instructed
    baseline_sim_scalar: float


def load_dataset() -> Dict[Subject, ProcessedSubjectData]:
    dataset_path = PROJECT_PATH / "dataset"

    results = {}
    for group in dataset_path.iterdir():
        for subj_folder in group.iterdir():
            subj = Subject(subj_folder.name, ExptCondition[group.name])
            results[subj] = ProcessedSubjectData(
                calibration=BlockContents(
                    features=np.load(subj_folder / "calibration" / "features.npy"),
                    one_hot_labels=np.load(subj_folder / "calibration" / "one_hot_labels.npy"),
                    probs=None,
                    game_lens=None,
                ),
                calibration_model=SVMNet().load_weights(subj_folder / "calibration" / "model_weights.pkl"),
                instructed=BlockContents(
                    features=np.load(subj_folder / "instructed_games" / "features.npy"),
                    one_hot_labels=np.load(subj_folder / "instructed_games" / "one_hot_labels.npy"),
                    game_lens=None,
                    probs=None,
                ),
                instructed_model=SVMNet().load_weights(subj_folder / "instructed_games" / "model_weights.pkl"),
                free=BlockContents(
                    features=np.load(subj_folder / "free_games" / "features.npy"),
                    one_hot_labels=np.load(subj_folder / "free_games" / "one_hot_labels.npy"),
                    game_lens=np.load(subj_folder / "free_games" / "game_lens.npy"),
                    probs=np.load(subj_folder / "free_games" / "probs.npy"),
                ),
            )
    return results


if __name__ == "__main__":
    load_dataset()
