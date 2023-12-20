Code for "**User Training with Error Augmentation for Electromyogram-based Gesture Classification**" by Yunus Bicer<sup>†</sup>, Niklas Smedemark-Margulies<sup>†</sup>, Basak Celik, Elifnur Sunger, Ryan Orendorff, Stephanie Naufel, Tales Imbiriba, Deniz Erdo˘gmus¸, Eugene Tunik, and Mathew Yarossi

# Setup and Usage

Use `make` to create python environment, install dependencies, and install `pre-commit` hooks.

To reproduce our experiments and analysis:

1. Unzip the included dataset (see below for dataset details):
```shell
unzip dataset.zip
```

2. Run analyses and generate figures:
```shell
source venv/bin/activate
python emg_feedback_user_training/main.py
```

## Dataset

Included in the repo is a file `dataset.zip` containing the dataset used for our analyses. 

Subjects are organized into folders based on the experiment group they were assigned to (`Control`, `Veridical`, and `Modified`)
Each subject's folder contains 3 subfolders: `calibration`, `instructed_games` and `free_games`, corresponding to 3 blocks of the experiment.
- `calibration` contains features, labels, and pre-trained weights for a model trained after this block.
- `instructed_games` contains the same as `calibration`.
- `free_games` contains the same, plus the length of each game in moves (since user planning and model decisions could affect these outcomes) and predicted probabilities computed for each move.

Random seed was not controlled when training models during the experiments; thus we include pre-trained model weights to ensure reproducibility.

Gestures are labeled with an integer, corresponding to these 9 possible classes: 
`["Up", "Thumb", "Right", "Pinch", "Down", "Fist", "Left", "Open", "Rest"]`

For details on feature extraction, see the paper.

# PDF

To read our paper, see: https://arxiv.org/pdf/2309.07289.pdf

# Citation

If you use this code or dataset, please cite our paper:
```bibtex
@article{bicer2023user,
    title={User Training with Error Augmentation for Electromyogram-based Gesture Classification},
    author={
        Bicer, Yunus and
        Smedemark-Margulies, Niklas and
        Celik, Basak and
        Sunger, Elifnur and
        Orendorff, Ryan and
        Naufel, Stephanie and
        Imbiriba, Tales and
        Erdogmus, Deniz and
        Tunik, Eugene and
        Yarossi, Mathew
    },
    journal={arXiv preprint arXiv:2309.07289},
    year={2023}
}
```
