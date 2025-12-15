import argparse
import random
import os
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import plotly.graph_objs as go
import nibabel as nib
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3DTranspose as KConv3DTranspose
from tensorflow.keras.utils import register_keras_serializable
import config
from Model import DiceCoefficientLoss
import plotly

from dash import Dash, dcc, html, Input, Output, State

import datetime
import json
import io
import base64
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
    image_data = img.dataobj
    image_data = np.asarray(image_data)

    image_data = image_data[34:194, 22:214, 13:141, ]
    image_data = normalize(image_data)
    # Reshaping the Input Image and Ground Truth(Mask)
    reshaped_image_data=image_data.reshape(1,160,192,128,4)

    print(reshaped_image_data.shape)
    print(type(reshaped_image_data))

    # Prediction - Our Segmentation
    Y_hat = model.predict(x=reshaped_image_data)
    Y_hat = np.argmax(Y_hat, axis=-1)
    print(f"Y_hat shape - {Y_hat.shape}")

    # Read the Input Image and Predicted Mask
    image = reshaped_image_data[0, :, :, :, 0].T
    mask = Y_hat[0].T

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

    return fig1, fig2

external_stylesheets = []
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# ---------- Global Theming ----------
BLACK = "#050509"
ORANGE = "#ff7a1a"
LIGHT_ORANGE = "#ffb347"
LIGHT_BLUE = "#52c7ff"
GLASS_BG = "rgba(15, 15, 25, 0.85)"

fig_1, fig_2 = input_image("test4d.nii.gz")

app.layout = html.Div(
    [
        # Custom CSS for glassmorphism + hover effects
        html.Style(
            """
            body {
                margin: 0;
                padding: 0;
                background: radial-gradient(circle at top left, #181820 0, #050509 45%, #000000 100%);
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: #f5f7ff;
            }
            .glass-shell {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: stretch;
                justify-content: stretch;
                background: radial-gradient(circle at top left, #181820 0, #050509 45%, #000000 100%);
            }
            .nav-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 18px 42px;
                position: sticky;
                top: 0;
                z-index: 10;
                backdrop-filter: blur(14px);
                background: linear-gradient(90deg, rgba(5,5,9,0.96), rgba(12,12,18,0.92));
                border-bottom: 1px solid rgba(255,255,255,0.04);
            }
            .nav-title {
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.9rem;
                color: #fefefe;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .nav-pill {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: linear-gradient(145deg, #ffb347, #ff7a1a);
                box-shadow: 0 0 12px rgba(255, 138, 76, 0.8);
            }
            .nav-links {
                display: flex;
                gap: 0.75rem;
                align-items: center;
            }
            .primary-btn, .ghost-btn {
                border-radius: 999px;
                padding: 9px 18px;
                border: none;
                font-size: 0.85rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                backdrop-filter: blur(10px);
                transition: all 0.18s ease-out;
            }
            .primary-btn {
                background: linear-gradient(135deg, #52c7ff, #b4fffd);
                color: #020307;
                box-shadow: 0 12px 30px rgba(82, 199, 255, 0.28);
            }
            .primary-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 16px 40px rgba(82, 199, 255, 0.4);
            }
            .ghost-btn {
                background: rgba(255, 255, 255, 0.02);
                color: #c0c4ff;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .ghost-btn:hover {
                background: rgba(255, 255, 255, 0.06);
                transform: translateY(-1px);
            }
            .main-grid {
                flex: 1;
                display: flex;
                flex-direction: row;
                padding: 22px 42px 40px;
                gap: 24px;
            }
            @media (max-width: 1024px) {
                .main-grid {
                    flex-direction: column;
                    padding: 18px 18px 28px;
                }
            }
            .glass-card {
                flex: 1;
                background: rgba(15, 15, 25, 0.88);
                border-radius: 28px;
                padding: 24px 26px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                box-shadow:
                    0 24px 60px rgba(0, 0, 0, 0.7),
                    0 0 0 1px rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(22px);
                display: flex;
                flex-direction: column;
                gap: 18px;
            }
            .hero-eyebrow {
                font-size: 0.78rem;
                letter-spacing: 0.26em;
                text-transform: uppercase;
                color: rgba(255, 255, 255, 0.6);
            }
            .hero-title {
                font-size: 2.2rem;
                line-height: 1.1;
                font-weight: 750;
                letter-spacing: 0.02em;
            }
            .hero-highlight {
                background: linear-gradient(120deg, #ffb347, #ff7a1a, #ffea82);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .hero-subtitle {
                font-size: 0.98rem;
                color: rgba(235, 238, 255, 0.82);
                max-width: 480px;
            }
            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            .chip {
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                color: rgba(244, 247, 255, 0.88);
            }
            .metrics-row {
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                margin-top: 8px;
            }
            .metric-pill {
                min-width: 120px;
                padding: 10px 14px;
                border-radius: 18px;
                background: radial-gradient(circle at top left, rgba(255, 122, 26, 0.34), rgba(0,0,0,0.5));
                border: 1px solid rgba(255, 186, 120, 0.35);
            }
            .metric-label {
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: rgba(255, 237, 217, 0.78);
            }
            .metric-value {
                font-size: 1.1rem;
                font-weight: 640;
                color: #fff5e6;
            }
            .upload-zone {
                border-radius: 22px;
                border: 1px dashed rgba(120, 215, 255, 0.85);
                background: radial-gradient(circle at top left, rgba(82,199,255,0.14), rgba(13,19,33,0.9));
                padding: 32px 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 10px;
                text-align: center;
                color: #e7f6ff;
            }
            .upload-caption {
                font-size: 0.8rem;
                color: rgba(203, 222, 255, 0.88);
            }
            .upload-note {
                font-size: 0.75rem;
                color: rgba(185, 205, 255, 0.7);
            }
            .results-grid {
                display: flex;
                flex-direction: row;
                gap: 18px;
                margin-top: 10px;
            }
            @media (max-width: 1200px) {
                .results-grid {
                    flex-direction: column;
                }
            }
            .result-panel {
                flex: 1;
                border-radius: 22px;
                background: radial-gradient(circle at top left, rgba(255,122,26,0.18), rgba(8,8,14,0.96));
                border: 1px solid rgba(255, 255, 255, 0.04);
                padding: 12px 12px 4px;
            }
            .result-heading {
                font-size: 0.8rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: rgba(255, 240, 220, 0.78);
                margin-bottom: 4px;
            }
            """
        ),
        dcc.Location(id="url", refresh=False),
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
            className="main-grid",
            children=[
                # Left: Hero / copy
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div("CLINICAL‑GRADE PIPELINE", className="hero-eyebrow"),
                        html.Div(
                            [
                                html.Span("3D Brain Tumor ", className="hero-title"),
                                html.Span("Segmentation", className="hero-title hero-highlight"),
                            ]
                        ),
                        html.Div(
                            "Visualize volumetric MRIs and overlay model predictions in an immersive 3D viewer. "
                            "Upload a NIfTI volume and explore every slice in one interactive canvas.",
                            className="hero-subtitle",
                        ),
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
                ),
                # Right: Sample visualization
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
                    ],
                ),
            ],
        ),
    ],
)

def parse_contents(filename):
    # Load and render a single saved NIfTI file by filename
    img, msk = input_image(filename)
    return html.Div([
        html.Div([
            dcc.Graph(id='g1', figure=img)
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='g2', figure=msk)
        ], className="six columns"),
    ], className="row")


@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
)
def update_output(image_contents, filenames):
    if not image_contents or not filenames:
        return

    saved_files = []
    os.makedirs(config.IMAGES_DATA_DIR, exist_ok=True)

    for content_str, original_name in zip(image_contents, filenames):
        ext = "".join(Path(original_name).suffixes) or ".nii"
        # Only accept NIfTI uploads
        if ext.lower() not in [".nii", ".nii.gz"]:
            return html.Div(
                "Please upload NIfTI files (.nii or .nii.gz).",
                style={"color": "red", "padding": "10px"},
            )

        data = content_str.split(",")[1]
        save_name = f"image_{len(saved_files)+1}{ext}"
        save_path = Path(config.IMAGES_DATA_DIR) / save_name
        with open(save_path, "wb") as fp:
            fp.write(base64.b64decode(data))
        saved_files.append(save_name)

    # Only display the first uploaded file for now
    return [parse_contents(saved_files[0])]

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/upload':
        return page_1_layout
    else:
        return index_page

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
