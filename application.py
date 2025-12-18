import argparse
import random
import os
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import plotly.graph_objs as go
import plotly.io as pio
import nibabel as nib
from skimage import measure
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3DTranspose as KConv3DTranspose
from tensorflow.keras.utils import register_keras_serializable
import config
from Model import DiceCoefficientLoss
import plotly
import dash

from dash import Dash, dcc, html, Input, Output, State

import datetime
import json
import io
import base64
import re
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

"""#### Loading model"""


@register_keras_serializable()
class CompatConv3DTranspose(KConv3DTranspose):
    """Compatibility wrapper to ignore unsupported 'groups' arg in saved H5."""

    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


def dice(y_true, y_pred):
    # computes the dice score on two tensors

    sum_p = K.sum(y_pred, axis=0)
    sum_r = K.sum(y_true, axis=0)
    sum_pr = K.sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + K.epsilon()) / (dice_denominator + K.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    # computes the dice for the whole tumor

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_whole = K.sum(y_true_f[..., 1:], axis=1)
    p_whole = K.sum(y_pred_f[..., 1:], axis=1)
    dice_whole = dice(y_whole, p_whole)
    return dice_whole


def dice_en_metric(y_true, y_pred):
    # computes the dice for the enhancing region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))

    # workaround for tf
    # y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    # p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)

    y_core = K.sum(y_true_f[:, 2:], axis=1)
    p_core = K.sum(y_pred_f[:, 2:], axis=1)
    dice_core = dice(y_core, p_core)
    return dice_core

def gen_dice_score(y_true, y_pred):
  y_true_f = K.reshape(y_true,shape=(-1,4))
  y_pred_f = K.reshape(y_pred,shape=(-1,4))
  sum_p=K.sum(y_pred_f,axis=-2)
  sum_r=K.sum(y_true_f,axis=-2)
  sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
  weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
  generalised_dice_numerator =2*K.sum(weights*sum_pr)
  generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
  generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
  return generalised_dice_score


def gen_dice_loss(y_true, y_pred):
  return 1 - gen_dice_score(y_true, y_pred)


# model_path=config.MODEL_PATH

custom_objs = {
    'gen_dice_loss': gen_dice_loss,
    'dice_whole_metric': dice_whole_metric,
    'dice_en_metric': dice_en_metric,
    'dice_core_metric': dice_core_metric,
    'Conv3DTranspose': CompatConv3DTranspose,
}
model = tf.keras.models.load_model('finalvalaug.h5', custom_objects=custom_objs, compile=False)

"""### Prediction """


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    return out


def normalize(image):
    img1 = itensity_normalize_one_volume(image[..., 0])
    img2 = itensity_normalize_one_volume(image[..., 1])
    img3 = itensity_normalize_one_volume(image[..., 2])
    img4 = itensity_normalize_one_volume(image[..., 3])
    img = np.stack((img1, img2, img3, img4), axis=-1)
    return img


def input_image(image):
    image_path = os.path.join(config.IMAGES_DATA_DIR, image)
    img = nib.load(image_path)
    affine = img.affine
    image_data = np.asarray(img.dataobj)

    image_data = image_data[34:194, 22:214, 13:141, ]
    image_data = normalize(image_data)
    reshaped_image_data=image_data.reshape(1,160,192,128,4)

    # Prediction - Our Segmentation
    Y_hat = model.predict(x=reshaped_image_data)
    Y_hat = np.argmax(Y_hat, axis=-1)

    # Read the Input Image and Predicted Mask
    image = reshaped_image_data[0, :, :, :, 0]  # keep (160,192,128)
    mask = Y_hat[0]  # (160,192,128)

    # For Colorscale
    pl_bone=[[0.0, 'rgb(0, 0, 0)'],
             [0.05, 'rgb(10, 10, 14)'],
             [0.1, 'rgb(21, 21, 30)'],
             [0.15, 'rgb(33, 33, 46)'],
             [0.2, 'rgb(44, 44, 62)'],
             [0.25, 'rgb(56, 55, 77)'],
             [0.3, 'rgb(66, 66, 92)'],
             [0.35, 'rgb(77, 77, 108)'],
             [0.4, 'rgb(89, 92, 121)'],
             [0.45, 'rgb(100, 107, 132)'],
             [0.5, 'rgb(112, 123, 143)'],
             [0.55, 'rgb(122, 137, 154)'],
             [0.6, 'rgb(133, 153, 165)'],
             [0.65, 'rgb(145, 169, 177)'],
             [0.7, 'rgb(156, 184, 188)'],
             [0.75, 'rgb(168, 199, 199)'],
             [0.8, 'rgb(185, 210, 210)'],
             [0.85, 'rgb(203, 221, 221)'],
             [0.9, 'rgb(220, 233, 233)'],
             [0.95, 'rgb(238, 244, 244)'],
             [1.0, 'rgb(255, 255, 255)']]

    r,c = image[0].shape
    n_slices = image.shape[0]
    height = (image.shape[0]-1) / 10
    grid = np.linspace(0, height, n_slices)
    slice_step = grid[1] - grid[0]

    rm,cm = mask[0].shape
    nm_slices = mask.shape[0]
    height_m = (mask.shape[0]-1) / 10
    grid_m = np.linspace(0, height_m, nm_slices)
    slice_step_m = grid_m[1] - grid_m[0]

    initial_slice = go.Surface(
                         z=height*np.ones((r,c)),
                         surfacecolor=np.flipud(image[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    initial_slice_m = go.Surface(
                         z=height_m*np.ones((rm,cm)),
                         surfacecolor=np.flipud(mask[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    frames = [go.Frame(data=[dict(type='surface',
                              z=(height-k*slice_step)*np.ones((r,c)),
                              surfacecolor=np.flipud(image[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, n_slices)]

    frames_m = [go.Frame(data=[dict(type='surface',
                              z=(height_m-k*slice_step_m)*np.ones((rm,cm)),
                              surfacecolor=np.flipud(mask[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, nm_slices)]

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [dict(steps = [dict(method= 'animate',
                                  args= [[f'frame{k+1}'],
                                        dict(mode= 'immediate', frame= dict(duration=20, redraw= True),transition=dict(duration= 0))
                                        ],
                                  label=f'{k+1}'
                                  )for k in range(n_slices)],
                    active=17,
                    transition= dict(duration= 0),
                    x=0, # slider starting position
                    y=0,
                    currentvalue=dict(font=dict(size=12),
                                      prefix='slice: ',
                                      visible=True,
                                      xanchor= 'center'
                                     ),
                   len=1.0) #slider length
               ]

    layout3d = dict(title_text='Slices of Brain in volumetric data: Input Image', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    layout3d_m = dict(title_text='Slices of Mask: Brain Segmentation', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    fig1 = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
    fig2 = go.Figure(data=[initial_slice_m], layout=layout3d_m, frames=frames_m)

    return fig1, fig2, image, mask, affine

external_stylesheets = []
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# ---------- Global Theming ----------
BLACK = "#050509"
ORANGE = "#ff7a1a"
LIGHT_ORANGE = "#ffb347"
LIGHT_BLUE = "#52c7ff"
GLASS_BG = "rgba(15, 15, 25, 0.85)"
MAX_FILE_MB = 400
EXPECTED_MASK_SHAPE = (160, 192, 128)

fig_1, fig_2, sample_vol, sample_mask, sample_affine = input_image("test4d.nii.gz")

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Large arrays must not be persisted; keep strictly in memory
        dcc.Store(id="vol-store", storage_type="memory"),
        dcc.Store(id="mask-store", storage_type="memory"),
        dcc.Store(id="holo-store", storage_type="memory"),
        # Small metadata can be storage-backed if desired (affine/case id only)
        dcc.Store(id="meta-store", storage_type="session"),
        dcc.Store(id="case-store", storage_type="session"),
        # Internal init flag stored in session so the one-time purge runs only once per session
        dcc.Store(id="init-flag", storage_type="session"),
        html.Div(id="page-content", className="glass-shell"),
    ]
)

index_page = html.Div(
    className="glass-shell",
    children=[
        html.Div(
            className="nav-bar",
            children=[
                html.Div(
                    className="nav-title",
                    children=[html.Div(className="nav-pill"), "NeuroGlass · 3D Tumor Segmentation"],
                ),
                html.Div(
                    className="nav-links",
                    children=[
                        dcc.Link(
                            html.Button("Upload Volume", className="primary-btn"),
                            href="/upload",
                            style={"textDecoration": "none"},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="landing-stack",
            children=[
                # Top: Full-width Clinical-Grade Pipeline banner
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div("CLINICAL‑GRADE", className="hero-eyebrow"),
                        html.Div(
                            [
                                html.Span("Clinical‑Grade 3D Brain Tumor ", className="hero-title"),
                                html.Span("Segmentation Pipeline", className="hero-title hero-highlight"),
                            ]
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "28px",
                                "alignItems": "flex-start",
                                "justifyContent": "space-between",
                                "flexWrap": "wrap",
                            },
                            children=[
                                # Left: Descriptive text (primary)
                                html.Div(
                                    children=[
                                        html.Div(
                                            "Automated 3D MRI tumor segmentation built for clarity and speed. "
                                            "Accepts 4‑channel NIfTI volumes and produces interactive visualizations "
                                            "to explore, validate, and understand tumor regions across the whole brain.",
                                            className="hero-subtitle",
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "rowGap": "10px",
                                        "flex": "2 1 600px",
                                        "minWidth": "420px",
                                        "maxWidth": "760px",
                                    },
                                ),
                                # Right: Technical summary (secondary)
                                html.Div(
                                    children=[
                                        html.Div(
                                            className="chip-row",
                                            children=[
                                                html.Div("4‑Channel MRI", className="chip"),
                                                html.Div("3D U‑Net", className="chip"),
                                                html.Div("Interactive viewer", className="chip"),
                                            ],
                                        ),
                                        html.Div(
                                            className="metrics-row",
                                            children=[
                                                html.Div(
                                                    className="metric-pill",
                                                    children=[
                                                        html.Div("Sample Volume", className="metric-label"),
                                                        html.Div("160 × 192 × 128", className="metric-value"),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="metric-pill",
                                                    children=[
                                                        html.Div("Tumor Regions", className="metric-label"),
                                                        html.Div("Whole · Core · Enh.", className="metric-value"),
                                                    ],
                                                ),
                                            ],
                                            style={"marginTop": "6px"},
                                        ),
                                        html.Div(
                                            style={"display": "flex", "gap": "0.75rem", "marginTop": "12px"},
                                            children=[
                                                dcc.Link(
                                                    html.Button("Upload your scan", className="primary-btn"),
                                                    href="/upload",
                                                    style={"textDecoration": "none"},
                                                ),
                                                dcc.Link(
                                                    html.Button("View sample", className="ghost-btn"),
                                                    href="/",
                                                    style={"textDecoration": "none"},
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "rowGap": "8px",
                                        "flex": "1 1 320px",
                                        "minWidth": "300px",
                                        "maxWidth": "420px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                # Below: Sample Prediction section (internal layout unchanged)
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div("Sample Prediction", className="result-heading"),
                        html.Div(
                            "Preview from built‑in reference volume (test4d.nii.gz). Use Upload to run your own.",
                            className="upload-note",
                        ),
                        html.Div(
                            className="results-grid",
                            children=[
                                html.Div(
                                    className="result-panel",
                                    children=[
                                        html.Div("Input MRI", className="result-heading"),
                                        dcc.Graph(id="sample_g1", figure=fig_1, config={"displaylogo": False}),
                                    ],
                                ),
                                html.Div(
                                    className="result-panel",
                                    children=[
                                        html.Div("Predicted Mask", className="result-heading"),
                                        dcc.Graph(id="sample_g2", figure=fig_2, config={"displaylogo": False}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

page_1_layout = html.Div(
    className="glass-shell",
    children=[
        html.Div(
            className="nav-bar",
            children=[
                html.Div(
                    className="nav-title",
                    children=[html.Div(className="nav-pill"), "NeuroGlass · Upload"],
                ),
                html.Div(
                    className="nav-links",
                    children=[
                        dcc.Link(
                            html.Button("Back to sample", className="ghost-btn"),
                            href="/",
                            style={"textDecoration": "none"},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="main-grid",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div("UPLOAD 3D VOLUME", className="hero-eyebrow"),
                        html.Div(
                            [
                                html.Span("Run inference on ", className="hero-title"),
                                html.Span("your own data", className="hero-title hero-highlight"),
                            ]
                        ),
                        html.Div(
                            "Drop a 4‑channel NIfTI volume (.nii or .nii.gz). "
                            "We will crop, normalize and run the pre‑trained model in real‑time.",
                            className="hero-subtitle",
                        ),
                        html.Div(
                            className="chip-row",
                            children=[
                                html.Div("Format: NIfTI", className="chip"),
                                html.Div("Shape: 160×192×128", className="chip"),
                            ],
                        ),
                        dcc.Upload(
                            id="upload-image",
                            className="upload-zone",
                            children=[
                                html.Div("Drop NIfTI volume here", style={"fontWeight": 600}),
                                html.Div("or click to browse your filesystem", className="upload-caption"),
                                html.Div(
                                    "Accepted: .nii / .nii.gz | Single 4‑channel volume",
                                    className="upload-note",
                                ),
                            ],
                            multiple=True,
                        ),
                        html.Div(
                            className="upload-feedback",
                            children=[
                                html.Div(id="upload-status", className="upload-note"),
                                html.Div(id="uploaded-files", className="upload-note"),
                                html.Div(id="upload-warning", className="upload-note upload-warning"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div("Prediction Viewer", className="result-heading"),
                        html.Div(
                            "After upload, the MRI and segmentation animation will appear below.",
                            className="upload-note",
                        ),
                        html.Div(id="output-image-upload", className="results-grid"),
                        html.Div(
                            className="glass-card",
                            children=[
                                html.Div("Export & Report", className="result-heading"),
                                html.Div(
                                    "Download predictions or a lightweight HTML report. "
                                    "Identifiers are removed; only the generated case ID is used.",
                                    className="upload-note",
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "10px",
                                        "alignItems": "center",
                                    },
                                    children=[
                                        html.Button("Download mask (NIfTI)", id="download-mask-btn", className="primary-btn"),
                                        html.Button("Export HTML report", id="download-report-btn", className="ghost-btn"),
                                        html.Div(id="export-status", className="upload-note"),
                                        dcc.Download(id="download-mask"),
                                        dcc.Download(id="download-report"),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="glass-card",
                            children=[
                                html.Div("2D Slice Viewer", className="result-heading"),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="plane-dropdown",
                                            options=[
                                                {"label": "Axial (Z)", "value": "axial"},
                                                {"label": "Coronal (Y)", "value": "coronal"},
                                                {"label": "Sagittal (X)", "value": "sagittal"},
                                            ],
                                            value="axial",
                                            clearable=False,
                                            style={"width": "220px"},
                                        ),
                                        dcc.Slider(
                                            id="slice-slider",
                                            min=0,
                                            max=127,
                                            step=1,
                                            value=0,
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                    style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                                ),
                                dcc.Graph(id="slice-view", config={"displaylogo": False}, style={"width": "100%"}),
                                html.Div(id="metrics-panel", className="upload-note"),
                            ],
                        ),
                        html.Div(
                            className="glass-card",
                            children=[
                                html.Div("3D View", className="result-heading"),
                                html.Div(
                                    "Hologram meshes derived from the predicted mask. Downsampled for speed.",
                                    className="upload-note",
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "12px",
                                        "alignItems": "center",
                                    },
                                    children=[
                                        dcc.Checklist(
                                            id="mesh-classes",
                                            options=[
                                                {"label": "Whole", "value": "whole"},
                                                {"label": "Core", "value": "core"},
                                                {"label": "Enhancing", "value": "enhancing"},
                                            ],
                                            value=["whole", "core", "enhancing"],
                                            inline=True,
                                            inputClassName="pill-check",
                                            labelStyle={"marginRight": "8px"},
                                        ),
                                        dcc.RadioItems(
                                            id="brain-toggle",
                                            options=[
                                                {"label": "Tumor only", "value": "tumor"},
                                                {"label": "Brain + Tumor", "value": "brain"},
                                            ],
                                            value="tumor",
                                            inline=True,
                                            labelStyle={"marginRight": "12px"},
                                        ),
                                        html.Div(
                                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                            children=[
                                                html.Span("Opacity", className="upload-note"),
                                                dcc.Slider(
                                                    id="mesh-opacity",
                                                    min=0.2,
                                                    max=1.0,
                                                    step=0.1,
                                                    value=0.6,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="holo-graph", config={"displaylogo": False}, style={"width": "100%", "height": "560px"}),
                                html.Div(id="holo-status", className="upload-note"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

def safe_case_id(name: str) -> str:
    """Sanitize case identifiers to avoid leaking PHI in filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return cleaned or "case"


def get_spacing(meta_json: str) -> Tuple[float, float, float]:
    spacing = (1.0, 1.0, 1.0)
    if not meta_json:
        return spacing
    meta, err = safe_meta(meta_json)
    if meta is None or err:
        return spacing
    aff = np.array(meta.get("affine", np.eye(4)))
    return tuple(np.abs(np.diag(aff)[:3]).tolist())


def arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_arr(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


def safe_b64_to_arr(s: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not s:
        return None, "Missing cached array."
    try:
        return b64_to_arr(s), None
    except Exception as exc:
        return None, f"Resetting cached array: {exc}"


def safe_meta(meta_json: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not meta_json:
        return None, "Missing metadata."
    try:
        return json.loads(meta_json), None
    except Exception as exc:
        return None, f"Resetting cached metadata: {exc}"


def downsample_volume(vol: np.ndarray, target_max_dim: int = 96) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    factors = []
    for dim in vol.shape[:3]:
        factors.append(max(1, int(np.ceil(dim / target_max_dim))))
    fz, fy, fx = factors
    ds = vol[::fz, ::fy, ::fx]
    return ds, (fz, fy, fx)


def make_mesh(region: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = measure.marching_cubes(region.astype(np.float32), level=0.5, spacing=spacing)
    return verts, faces


def make_brain_surface(vol: np.ndarray, spacing: Tuple[float, float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    brain_vol = vol[..., 0]
    brain_vol = np.where(np.isnan(brain_vol), 0, brain_vol)
    positives = brain_vol[brain_vol > 0]
    if positives.size == 0:
        return None
    thr = float(np.percentile(positives, 60))
    brain_mask = (brain_vol > thr).astype(np.uint8)
    if brain_mask.sum() < 500:
        return None
    return make_mesh(brain_mask, spacing)


def build_hologram_figure(
    vol: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    enabled_classes: List[str],
    opacity: float,
    mode: str,
) -> Tuple[go.Figure, str]:
    base_trace = []
    status = []

    # Downsample to keep marching cubes light
    mask_ds, (fz, fy, fx) = downsample_volume(mask)
    vol_ds, _ = downsample_volume(vol)
    spacing_ds = (spacing[0] * fz, spacing[1] * fy, spacing[2] * fx)

    regions = {
        "whole": (mask_ds > 0, "#52c7ff"),
        "core": (np.isin(mask_ds, [2, 3]), "#ff7a1a"),
        "enhancing": (mask_ds == 3, "#ff3366"),
    }
    for key, (region, color) in regions.items():
        if key not in enabled_classes:
            continue
        if region.sum() < 50:
            status.append(f"No voxels for {key}.")
            continue
        try:
            verts, faces = make_mesh(region, spacing_ds)
            base_trace.append(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=opacity,
                    name=key.title(),
                    flatshading=True,
                )
            )
        except Exception as exc:
            status.append(f"{key} mesh failed: {exc}")

    # Optional brain surface
    if mode == "brain" and vol_ds.size > 0:
        brain_mesh = make_brain_surface(vol_ds, spacing_ds)
        if brain_mesh:
            verts, faces = brain_mesh
            base_trace.insert(
                0,
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightgray",
                    opacity=0.18,
                    name="Brain surface",
                    flatshading=True,
                ),
            )
        else:
            status.append("Brain surface skipped (low signal).")

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom"),
        title="3D Hologram View",
    )
    fig = go.Figure(data=base_trace, layout=layout)
    if not base_trace:
        fig.add_annotation(text="No meshes to show.", showarrow=False)
        status.append("Nothing to render.")
    fig.update_layout(uirevision="holo")
    status_text = " | ".join(status) if status else "Ready."
    return fig, status_text


def build_slice_html(vol: np.ndarray, mask: np.ndarray, plane: str, spacing: Tuple[float, float, float]) -> str:
    axis_len = vol.shape[2] if plane == "axial" else vol.shape[1] if plane == "coronal" else vol.shape[0]
    idx = axis_len // 2
    vol_slice, mask_slice = slice_by_plane(vol, mask, plane, idx)
    fig = make_slice_figure(vol_slice, mask_slice)
    fig.update_layout(width=420, height=420, title=f"{plane.title()} slice @ {idx}")
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def build_report_html(case_id: str, vol: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float, float]) -> str:
    metrics = compute_metrics(mask, spacing)
    rows = "".join(
        f"<tr><td>{c['class']}</td><td>{c['voxels']}</td><td>{c['ml']}</td></tr>"
        for c in metrics["classes"]
    )
    slices_html = "".join(
        build_slice_html(vol, mask, plane, spacing) for plane in ["axial", "coronal", "sagittal"]
    )
    html_doc = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Segmentation Report - {case_id}</title>
    </head>
    <body style="font-family:Arial,sans-serif; background:#0b0b14; color:#e8ecf1;">
        <h2>Segmentation Report</h2>
        <p>Case ID: {case_id}</p>
        <p>Backend/runtime: Dash UI (pre-computed)</p>
        <h3>Volumes (ml)</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
            <tr><th>Class</th><th>Voxels</th><th>Volume (ml)</th></tr>
            {rows}
        </table>
        <h3>Key slices with overlays</h3>
        {slices_html}
    </body>
    </html>
    """
    return html_doc


def parse_contents(filename: str, case_id: str) -> Tuple[Any, str, str, str, str]:
    """Load, run prediction, and package data for UI stores."""
    warning = ""
    fig1, fig2, vol, msk, affine = input_image(filename)
    if vol.shape != EXPECTED_MASK_SHAPE or msk.shape != EXPECTED_MASK_SHAPE:
        warning = (
            f"Warning: processed volume/mask shape {vol.shape} differs from expected {EXPECTED_MASK_SHAPE}. "
            "Visualization may be limited."
        )
    children = html.Div(
        [
            html.Div([dcc.Graph(id="g1", figure=fig1)], className="six columns"),
            html.Div([dcc.Graph(id="g2", figure=fig2)], className="six columns"),
        ],
        className="row",
    )
    return (
        children,
        arr_to_b64(vol),
        arr_to_b64(msk),
        json.dumps({"affine": affine.tolist()}),
        warning,
    )


@app.callback(
    Output("output-image-upload", "children"),
    Output("vol-store", "data"),
    Output("mask-store", "data"),
    Output("meta-store", "data"),
    Output("case-store", "data"),
    Output("upload-status", "children"),
    Output("uploaded-files", "children"),
    Output("upload-warning", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def update_output(image_contents, filenames):
    if not image_contents or not filenames:
        status = "Awaiting upload..."
        file_list = html.Div("No files uploaded yet.")
        return None, None, None, None, None, status, file_list, ""

    saved_files = []
    original_names = []
    warnings = []
    os.makedirs(config.IMAGES_DATA_DIR, exist_ok=True)

    for content_str, original_name in zip(image_contents, filenames):
        ext = "".join(Path(original_name).suffixes) or ".nii"
        if ext.lower() not in [".nii", ".nii.gz"]:
            msg = "Please upload NIfTI files (.nii or .nii.gz)."
            return (
                html.Div(msg, style={"color": "red", "padding": "10px"}),
                None,
                None,
                None,
                None,
                "Invalid file type.",
                html.Div("No files saved. Please upload .nii or .nii.gz."),
                msg,
            )

        data = content_str.split(",")[1]
        size_mb = len(data) * 0.75 / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            warnings.append(f"File {original_name} is {size_mb:.1f} MB; this may be too large to render.")

        save_name = f"image_{len(saved_files)+1}{ext}"
        save_path = Path(config.IMAGES_DATA_DIR) / save_name
        with open(save_path, "wb") as fp:
            fp.write(base64.b64decode(data))
        saved_files.append(save_name)
        original_names.append(original_name)

    case_id = safe_case_id(f"case_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    children, vol_b64, mask_b64, meta_json, inference_warning = parse_contents(saved_files[0], case_id)
    if inference_warning:
        warnings.append(inference_warning)
    status = f"Uploaded {len(saved_files)} file(s) successfully."
    file_list = html.Div(
        [
            html.Div(
                [
                    html.Span("*", className="file-dot"),
                    html.Span(orig, className="file-name"),
                ],
                className="file-pill",
            )
            for orig in original_names
        ]
    )
    warning_block = html.Div([html.Div(w) for w in warnings]) if warnings else ""
    return [children], vol_b64, mask_b64, meta_json, case_id, status, file_list, warning_block

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/upload':
        return page_1_layout
    else:
        return index_page


# ----- Slice viewer & metrics -----
def slice_by_plane(vol: np.ndarray, mask: np.ndarray, plane: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if plane == "axial":
        vol_slice = vol[:, :, idx]
        mask_slice = mask[:, :, idx]
    elif plane == "coronal":
        vol_slice = vol[:, idx, :]
        mask_slice = mask[:, idx, :]
    else:  # sagittal
        vol_slice = vol[idx, :, :]
        mask_slice = mask[idx, :, :]
    return vol_slice, mask_slice


def make_slice_figure(vol_slice: np.ndarray, mask_slice: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=vol_slice, colorscale="gray", showscale=False))
    fig.add_trace(
        go.Heatmap(
            z=np.ma.masked_where(mask_slice == 0, mask_slice),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(82,199,255,0.65)"]],
            showscale=False,
            opacity=0.8,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def compute_metrics(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, Any]:
    voxel_vol_ml = float(np.prod(spacing) / 1000.0)
    metrics = {"voxel_volume_ml": voxel_vol_ml, "classes": []}
    for cls in sorted(np.unique(mask)):
        if cls == 0:
            continue
        vox = int((mask == cls).sum())
        metrics["classes"].append({"class": int(cls), "voxels": vox, "ml": round(vox * voxel_vol_ml, 3)})
    return metrics


@app.callback(
    Output("slice-view", "figure"),
    Output("slice-slider", "max"),
    Output("slice-slider", "value"),
    Output("metrics-panel", "children"),
    Input("vol-store", "data"),
    Input("mask-store", "data"),
    Input("meta-store", "data"),
    Input("plane-dropdown", "value"),
    Input("slice-slider", "value"),
)
def update_slice(vol_b64, mask_b64, meta_json, plane, slice_idx):
    vol, vol_err = safe_b64_to_arr(vol_b64)
    mask, mask_err = safe_b64_to_arr(mask_b64)
    if vol is None or mask is None:
        msg = vol_err or mask_err or "Awaiting upload or data reset due to incompatible cached content."
        return go.Figure(), 0, 0, msg

    # Validate shape compatibility
    if vol.ndim != 3 or mask.ndim != 3 or vol.shape != mask.shape:
        return go.Figure(), 0, 0, "Stored data schema mismatch; please re-upload."

    spacing = get_spacing(meta_json)

    axis_len = vol.shape[2] if plane == "axial" else vol.shape[1] if plane == "coronal" else vol.shape[0]
    if slice_idx is None or not isinstance(slice_idx, int) or slice_idx >= axis_len or slice_idx < 0:
        slice_idx = axis_len // 2

    vol_slice, mask_slice = slice_by_plane(vol, mask, plane, slice_idx)
    fig = make_slice_figure(vol_slice, mask_slice)

    metrics = compute_metrics(mask, spacing)
    metrics_text = [html.Div(f"Voxel volume: {metrics['voxel_volume_ml']:.4f} ml")]
    metrics_text += [html.Div(f"Class {c['class']}: {c['voxels']} voxels | {c['ml']} ml") for c in metrics["classes"]]
    if vol_err or mask_err:
        metrics_text.append(html.Div(vol_err or mask_err, style={"color": "orange"}))

    return fig, axis_len - 1, slice_idx, metrics_text


@app.callback(
    Output("holo-graph", "figure"),
    Output("holo-status", "children"),
    Input("vol-store", "data"),
    Input("mask-store", "data"),
    Input("meta-store", "data"),
    Input("mesh-classes", "value"),
    Input("mesh-opacity", "value"),
    Input("brain-toggle", "value"),
)
def update_hologram(vol_b64, mask_b64, meta_json, classes, opacity, mode):
    vol, vol_err = safe_b64_to_arr(vol_b64)
    mask, mask_err = safe_b64_to_arr(mask_b64)

    status_msgs = []
    if vol_err:
        status_msgs.append(vol_err)
    if mask_err:
        status_msgs.append(mask_err)

    if vol is None or mask is None:
        fig = go.Figure()
        fig.add_annotation(text="Awaiting upload.", showarrow=False)
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        return fig, "Awaiting upload or cache reset."

    if vol.ndim != 3 or mask.ndim != 3 or vol.shape != mask.shape:
        fig = go.Figure()
        fig.add_annotation(text="Shape mismatch; please re-upload.", showarrow=False)
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        return fig, "Stored data schema mismatch; please re-upload."

    spacing = get_spacing(meta_json)
    enabled_classes = classes or []
    fig, base_status = build_hologram_figure(
        vol=vol,
        mask=mask,
        spacing=spacing,
        enabled_classes=enabled_classes,
        opacity=opacity if opacity else 0.6,
        mode=mode if mode else "tumor",
    )
    msg = base_status
    if status_msgs:
        msg = " | ".join([base_status] + status_msgs if base_status else status_msgs)
    return fig, msg or "Ready."


@app.callback(
    Output("download-mask", "data"),
    Output("download-report", "data"),
    Output("export-status", "children"),
    Input("download-mask-btn", "n_clicks"),
    Input("download-report-btn", "n_clicks"),
    State("mask-store", "data"),
    State("vol-store", "data"),
    State("meta-store", "data"),
    State("case-store", "data"),
    prevent_initial_call=True,
)
def handle_exports(mask_clicks, report_clicks, mask_b64, vol_b64, meta_json, case_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    case_id = safe_case_id(case_data or "case")

    if trigger == "download-mask-btn":
        # Defensive checks for mask presence and shape
        mask, err = safe_b64_to_arr(mask_b64)
        if mask is None:
            return dash.no_update, dash.no_update, err or "Mask unavailable; please re-upload."
        if mask.ndim != 3:
            return dash.no_update, dash.no_update, "Mask is not 3D; export aborted."

        # Affine from metadata (fallback to identity). Ensure 4x4
        affine = np.eye(4)
        meta, _ = safe_meta(meta_json) if meta_json else (None, None)
        if meta and isinstance(meta, dict) and "affine" in meta:
            try:
                aff = np.array(meta["affine"], dtype=float)
                if aff.shape == (4, 4):
                    affine = aff
            except Exception:
                pass

        # Save NIfTI to a temporary .nii.gz file (Windows-safe), then stream bytes
        import tempfile, os
        import pathlib
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
            tmp_path = tmp_file.name
            tmp_file.close()  # close handle before nib.save on Windows
            img = nib.Nifti1Image(mask.astype(np.uint8, copy=False), affine)
            nib.save(img, tmp_path)
            with open(tmp_path, "rb") as f:
                content = f.read()
        finally:
            if tmp_file is not None:
                try:
                    os.remove(tmp_file.name)
                except Exception:
                    pass

        filename = f"{case_id}_pred_mask.nii.gz"
        data = dcc.send_bytes(lambda b: b.write(content), filename)
        return data, dash.no_update, "Mask download ready."

    if trigger == "download-report-btn":
        vol, v_err = safe_b64_to_arr(vol_b64)
        mask, m_err = safe_b64_to_arr(mask_b64)
        if vol is None or mask is None:
            msg = v_err or m_err or "Data unavailable; please re-upload."
            return dash.no_update, dash.no_update, msg
        spacing = get_spacing(meta_json)
        html_report = build_report_html(case_id, vol, mask, spacing)
        data = dict(content=html_report, filename=f"{case_id}_report.html", type="text/html")
        return dash.no_update, data, "Report download ready."

    return dash.no_update, dash.no_update, dash.no_update

# One-time clientside init to purge any persisted Store content from older app versions.
app.clientside_callback(
    """
    function(pathname, init) {
        const first = init !== "done";
        if (first) {
            // Purge only small metadata keys; big Stores are memory-only
            try {
                ['meta-store','case-store'].forEach(function(id) {
                    try { window.sessionStorage.removeItem(id); } catch(e) {}
                    try { window.localStorage.removeItem(id); } catch(e) {}
                });
            } catch(e) {}
        }
        // clear_data flags mirror 'first' so stores reset on initial load only
        return ["done", first, first, first, first, first];
    }
    """,
    Output('init-flag', 'data'),
    Output('vol-store', 'clear_data'),
    Output('mask-store', 'clear_data'),
    Output('meta-store', 'clear_data'),
    Output('case-store', 'clear_data'),
    Output('holo-store', 'clear_data'),
    Input('url', 'pathname'),
    State('init-flag', 'data'),
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Dash app")
    default_port = int(os.environ.get("PORT", 8060))
    default_host = os.environ.get("HOST", "0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port, help="Port to run the server on")
    parser.add_argument("--host", type=str, default=default_host, help="Host address to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    # Dash 3 deprecates run_server in favor of run
    app.run(debug=args.debug, host=args.host, port=args.port)
