from pathlib import Path

import numpy as np
import plotly.io as pio

PROJECT_PATH = Path(__file__).resolve().parent.parent
GESTURE_NAMES = ["Up", "Thumb", "Right", "Pinch", "Down", "Fist", "Left", "Open", "Rest"]
NUM_CLASSES = len(GESTURE_NAMES)
DATA_DTYPE = np.float32

pio.templates.default = "simple_white"

IMG_HEIGHT = 600
IMG_WIDTH = 800
IMG_SCALE = 2.0
FONT_SIZE = 20


def savefig(
    fig,
    output_path,
    basename,
    height=IMG_HEIGHT,
    width=IMG_WIDTH,
    scale=IMG_SCALE,
    font_size=FONT_SIZE,
    html=True,
    png=True,
    pdf=False,
    legend_top=True,
):
    fig.update_layout(font_size=font_size)
    fig.update_layout(legend_tracegroupgap=0)  # removes additional spacing between legend groups
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="center", x=0.5))

    if html:
        fig.write_html(output_path / f"{basename}.html", include_plotlyjs="cdn", include_mathjax="cdn")
    if png:
        fig.write_image(output_path / f"{basename}.png", height=height, width=width, scale=scale)
    if pdf:
        fig.write_image(output_path / f"{basename}.pdf", height=height, width=width, scale=scale)
