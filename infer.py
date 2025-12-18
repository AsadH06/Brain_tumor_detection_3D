"""
Inference CLI with strict validation, ONNX export, and batch processing.

Features:
- Backend toggle: TensorFlow (default) or ONNX Runtime for faster inference.
- ONNX export utility (tf.keras -> ONNX) with optional auto-export when missing.
- Batch mode: process multiple patients in a folder, save masks + summary CSV.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose as KConv3DTranspose
from tensorflow.keras.utils import register_keras_serializable

import config

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

try:
    import tf2onnx
except ImportError:
    tf2onnx = None  # type: ignore

# ---- Defaults ----
DEFAULT_TF_MODEL = Path(getattr(config, "TF_MODEL_PATH", "finalvalaug.h5"))
DEFAULT_ONNX_MODEL = Path(getattr(config, "ONNX_MODEL_PATH", "finalvalaug.onnx"))
DEFAULT_BACKEND = getattr(config, "INFERENCE_BACKEND", "tf")
DEFAULT_BATCH_INPUT_DIR = Path(getattr(config, "BATCH_INPUT_DIR", "BrainTumorData/imagesTest"))
DEFAULT_BATCH_OUTPUT_DIR = Path(getattr(config, "BATCH_OUTPUT_DIR", "outputs"))


@dataclass
class CaseSpec:
    case_id: str
    input_4d: Optional[Path]
    modalities: Optional[Dict[str, Path]]
    root: Path


# ---- Model helpers ----
@register_keras_serializable()
class CompatConv3DTranspose(KConv3DTranspose):
    """Compatibility wrapper to ignore unsupported 'groups' arg in saved H5."""

    @classmethod
    def from_config(cls, cfg):
        cfg.pop("groups", None)
        return super().from_config(cfg)


def load_seg_model(model_path: Path):
    custom_objs = {"Conv3DTranspose": CompatConv3DTranspose}
    return tf.keras.models.load_model(str(model_path), custom_objects=custom_objs, compile=False)


class TfEngine:
    backend = "tf"

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"TensorFlow model not found: {model_path}")
        self.model_path = model_path
        self.model = load_seg_model(model_path)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        logits = self.model.predict(batch)
        return np.argmax(logits, axis=-1)


class OnnxEngine:
    backend = "onnx"

    def __init__(self, model_path: Path):
        if ort is None:
            raise ImportError("onnxruntime is not installed; pip install onnxruntime.")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        providers = ort.get_available_providers()
        # Prefer GPU if present, otherwise CPU.
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: batch.astype(np.float32)})
        return np.argmax(outputs[0], axis=-1)


def export_to_onnx(tf_model_path: Path, onnx_output: Path, opset: int = 13, overwrite: bool = False):
    if tf2onnx is None:
        raise ImportError("tf2onnx is not installed; pip install tf2onnx to export ONNX.")
    if not tf_model_path.exists():
        raise FileNotFoundError(f"Cannot export; TF model not found: {tf_model_path}")
    if onnx_output.exists() and not overwrite:
        raise FileExistsError(f"ONNX file already exists: {onnx_output}. Use --force-export to overwrite.")

    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    model = load_seg_model(tf_model_path)
    # Input signature matches training window (None, 160, 192, 128, 4)
    signature = (tf.TensorSpec((None, 160, 192, 128, 4), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        model,
        input_signature=signature,
        opset=opset,
        output_path=str(onnx_output),
    )
    print(f"Exported ONNX model to {onnx_output}")


# ---- Validation helpers ----
def _error(msg: str):
    raise ValueError(msg)


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if not path.exists():
        _error(f"File not found: {path}")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data


def validate_modalities(modality_paths: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate four modality files (t1, t1ce, t2, flair)."""
    required = ["t1", "t1ce", "t2", "flair"]
    for key in required:
        if key not in modality_paths or modality_paths[key] is None:
            _error(f"Missing {key}")

    imgs = {}
    datas = {}
    for key, path in modality_paths.items():
        img, data = _load_nifti(path)
        if np.isnan(data).any():
            _error(f"NaNs detected in {key}")
        imgs[key] = img
        datas[key] = data

    shapes = {k: v.shape for k, v in datas.items()}
    if len(set(shapes.values())) != 1:
        _error(f"Modality shapes differ: {shapes}")

    affines = {k: v.affine for k, v in imgs.items()}
    ref_aff = affines["t1"]
    for k, aff in affines.items():
        if not np.allclose(ref_aff, aff, atol=1e-3):
            _error(f"Affine mismatch for {k}")

    stacked = np.stack([datas["t1"], datas["t1ce"], datas["t2"], datas["flair"]], axis=-1)
    return ref_aff, stacked


def validate_single_4d(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img, data = _load_nifti(path)
    if data.ndim != 4 or data.shape[-1] != 4:
        _error(f"Expected 4D volume with 4 channels, got shape {data.shape}")
    if np.isnan(data).any():
        _error("NaNs detected in 4D input")
    return img.affine, data


# ---- Pre / Post processing ----
def itensity_normalize_one_volume(volume: np.ndarray) -> np.ndarray:
    pixels = volume[volume > 0]
    if pixels.size == 0:
        return np.zeros_like(volume)
    mean = pixels.mean()
    std = pixels.std() if pixels.std() > 0 else 1.0
    return (volume - mean) / std


def normalize(image: np.ndarray) -> np.ndarray:
    channels = [itensity_normalize_one_volume(image[..., i]) for i in range(4)]
    return np.stack(channels, axis=-1)


def crop_to_training_window(img4d: np.ndarray) -> np.ndarray:
    return img4d[34:194, 22:214, 13:141, :]


def save_nifti(array: np.ndarray, affine: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Keep largest component per class >0."""
    out = np.zeros_like(mask)
    for cls in np.unique(mask):
        if cls == 0:
            continue
        cls_mask = (mask == cls).astype(np.uint8)
        labeled, n = ndi.label(cls_mask)
        if n == 0:
            continue
        sizes = ndi.sum(cls_mask, labeled, index=range(1, n + 1))
        largest = (np.argmax(sizes) + 1)
        out[labeled == largest] = cls
    return out


def remap_labels(mask: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "0124":
        mask = mask.copy()
        mask[mask == 3] = 4
    return mask


# ---- Inference helpers ----
def build_engine(backend: str, tf_model: Path, onnx_model: Path, opset: int, allow_export: bool) -> object:
    backend = backend.lower()
    if backend == "tf":
        return TfEngine(tf_model)
    if backend == "onnx":
        if not onnx_model.exists():
            if not allow_export:
                raise FileNotFoundError(f"ONNX model missing: {onnx_model}")
            export_to_onnx(tf_model, onnx_model, opset=opset, overwrite=True)
        return OnnxEngine(onnx_model)
    raise ValueError(f"Unsupported backend: {backend}")


def prepare_volume(
    input_4d: Optional[Path],
    t1: Optional[Path],
    t1ce: Optional[Path],
    t2: Optional[Path],
    flair: Optional[Path],
    save_pre: bool,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[Path], str, str]:
    if input_4d:
        affine, volume = validate_single_4d(input_4d)
        input_kind = "4d"
        input_source = str(input_4d)
    else:
        affine, volume = validate_modalities({"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair})
        input_kind = "modalities"
        input_source = ";".join([str(t1), str(t1ce), str(t2), str(flair)])

    volume = crop_to_training_window(volume)
    volume = normalize(volume)

    pre_path = None
    if save_pre:
        pre_path = output_dir / "preprocessed.nii.gz"
        save_nifti(volume, affine, pre_path)

    return affine, volume, pre_path, input_kind, input_source


def run_single_case(
    engine,
    spec: CaseSpec,
    output_dir: Path,
    cleanup: bool,
    label_scheme: str,
    save_pre: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    affine, volume, pre_path, input_kind, input_source = prepare_volume(
        input_4d=spec.input_4d,
        t1=spec.modalities.get("t1") if spec.modalities else None,
        t1ce=spec.modalities.get("t1ce") if spec.modalities else None,
        t2=spec.modalities.get("t2") if spec.modalities else None,
        flair=spec.modalities.get("flair") if spec.modalities else None,
        save_pre=save_pre,
        output_dir=output_dir,
    )

    reshaped = volume.reshape(1, 160, 192, 128, 4).astype(np.float32)
    start = time.time()
    y_hat = engine.predict(reshaped)[0]
    infer_seconds = time.time() - start

    if cleanup:
        y_hat = cleanup_mask(y_hat)
    y_hat = remap_labels(y_hat, label_scheme)

    mask_path = output_dir / "pred_mask.nii.gz"
    save_nifti(y_hat.astype(np.uint8), affine, mask_path)

    print(f"[{spec.case_id}] Saved predicted mask to: {mask_path}")

    return {
        "case_id": spec.case_id,
        "backend": engine.backend,
        "model_path": str(engine.model_path),
        "input_kind": input_kind,
        "input_source": input_source,
        "mask_path": str(mask_path),
        "preprocessed_path": str(pre_path) if pre_path else "",
        "label_scheme": label_scheme,
        "cleanup": bool(cleanup),
        "inference_seconds": round(infer_seconds, 4),
        "status": "success",
        "message": "",
    }


# ---- Batch helpers ----
def find_modalities(case_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for file in case_dir.iterdir():
        if not file.is_file() or not is_nifti(file):
            continue
        name = file.name.lower()
        if "flair" in name and "flair" not in mapping:
            mapping["flair"] = file
        elif "t1ce" in name and "t1ce" not in mapping:
            mapping["t1ce"] = file
        elif ("t1" in name or "t1w" in name) and "t1ce" not in name and "t1" not in mapping:
            mapping["t1"] = file
        elif "t2" in name and "t2" not in mapping:
            mapping["t2"] = file
    if len(mapping) == 4:
        return mapping
    return {}


def discover_cases(patients_dir: Path) -> List[CaseSpec]:
    cases: List[CaseSpec] = []

    # Direct NIfTI files inside patients_dir -> treated as single 4D cases
    for file in sorted(patients_dir.iterdir()):
        if file.is_file() and is_nifti(file):
            cases.append(CaseSpec(case_id=file.stem, input_4d=file, modalities=None, root=file.parent))

    # Subdirectories with either modalities or a single 4D NIfTI
    for sub in sorted(p for p in patients_dir.iterdir() if p.is_dir()):
        mods = find_modalities(sub)
        if mods:
            cases.append(CaseSpec(case_id=sub.name, input_4d=None, modalities=mods, root=sub))
            continue
        nifti_files = [f for f in sub.iterdir() if f.is_file() and is_nifti(f)]
        if nifti_files:
            cases.append(CaseSpec(case_id=sub.name, input_4d=nifti_files[0], modalities=None, root=sub))

    return cases


def run_batch(
    engine,
    patients_dir: Path,
    output_root: Path,
    cleanup: bool,
    label_scheme: str,
    save_pre: bool,
    limit: Optional[int],
    summary_name: str,
):
    cases = discover_cases(patients_dir)
    if not cases:
        _error(f"No NIfTI files found under {patients_dir}")

    if limit:
        cases = cases[:limit]

    rows: List[Dict[str, object]] = []
    for spec in cases:
        try:
            out_dir = output_root / spec.case_id
            result = run_single_case(engine, spec, out_dir, cleanup, label_scheme, save_pre)
        except Exception as exc:
            result = {
                "case_id": spec.case_id,
                "backend": engine.backend,
                "model_path": str(engine.model_path),
                "input_kind": "unknown",
                "input_source": str(spec.root),
                "mask_path": "",
                "preprocessed_path": "",
                "label_scheme": label_scheme,
                "cleanup": bool(cleanup),
                "inference_seconds": 0.0,
                "status": "failed",
                "message": str(exc),
            }
            print(f"[{spec.case_id}] Failed: {exc}")

        rows.append(result)

    df = pd.DataFrame(rows)
    summary_path = output_root / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"Wrote batch summary to {summary_path}")


# ---- CLI ----
def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D tumor segmentation inference with ONNX + batch options.")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input-4d", help="Path to a single 4D NIfTI (.nii/.nii.gz) with 4 channels.")
    parser.add_argument("--t1", help="Path to T1 NIfTI (.nii/.nii.gz).")
    parser.add_argument("--t1ce", help="Path to T1ce NIfTI (.nii/.nii.gz).")
    parser.add_argument("--t2", help="Path to T2 NIfTI (.nii/.nii.gz).")
    parser.add_argument("--flair", help="Path to FLAIR NIfTI (.nii/.nii.gz).")

    parser.add_argument("--out", required=True, help="Output directory to store prediction and intermediates.")
    parser.add_argument("--backend", choices=["tf", "onnx"], default=DEFAULT_BACKEND, help="Inference backend.")
    parser.add_argument("--tf-model", default=str(DEFAULT_TF_MODEL), help="Path to trained model H5.")
    parser.add_argument("--onnx-model", default=str(DEFAULT_ONNX_MODEL), help="Path to ONNX model.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version for export.")
    parser.add_argument("--export-onnx", help="Export TF model to ONNX and exit (path to write).")
    parser.add_argument("--force-export", action="store_true", help="Overwrite ONNX file if it exists.")
    parser.add_argument("--cleanup", action="store_true", help="Enable basic post-processing (largest component per class).")
    parser.add_argument("--label-scheme", choices=["0123", "0124"], default="0123", help="Remap labels: keep 0/1/2/3 or map 3->4.")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save cropped+normalized 4D volume to disk.")

    parser.add_argument("--patients-dir", help="Directory containing patient subfolders or 4D volumes for batch mode.")
    parser.add_argument("--limit", type=int, help="Limit number of patients to process in batch mode.")
    parser.add_argument("--summary-name", default="summary.csv", help="Filename for batch summary CSV.")

    args = parser.parse_args()

    if args.export_onnx:
        return args

    if not args.patients_dir:
        # Enforce modality set if no 4D input given
        if not args.input_4d:
            missing = [k for k in ["t1", "t1ce", "t2", "flair"] if getattr(args, k) is None]
            if missing:
                _error(f"Missing modalities: {', '.join(missing)}. Provide --input-4d or all four modalities.")

    return args


def main():
    args = parse_args()
    out_dir = Path(args.out)

    # Handle export-only mode
    if args.export_onnx:
        export_to_onnx(Path(args.tf_model), Path(args.export_onnx), opset=args.opset, overwrite=bool(args.force_export))
        return

    backend = args.backend.lower()
    tf_model_path = Path(args.tf_model)
    onnx_model_path = Path(args.onnx_model)

    engine = build_engine(
        backend=backend,
        tf_model=tf_model_path,
        onnx_model=onnx_model_path,
        opset=int(args.opset),
        allow_export=True,
    )

    if args.patients_dir:
        patients_dir = Path(args.patients_dir)
        run_batch(
            engine=engine,
            patients_dir=patients_dir,
            output_root=out_dir,
            cleanup=bool(args.cleanup),
            label_scheme=args.label_scheme,
            save_pre=bool(args.save_preprocessed),
            limit=int(args.limit) if args.limit else None,
            summary_name=args.summary_name,
        )
        return

    # Single-case inference
    spec = CaseSpec(
        case_id=Path(args.input_4d).stem if args.input_4d else Path(args.t1).stem,
        input_4d=Path(args.input_4d) if args.input_4d else None,
        modalities={
            "t1": Path(args.t1) if args.t1 else None,
            "t1ce": Path(args.t1ce) if args.t1ce else None,
            "t2": Path(args.t2) if args.t2 else None,
            "flair": Path(args.flair) if args.flair else None,
        }
        if not args.input_4d
        else None,
        root=Path("."),
    )
    result = run_single_case(
        engine=engine,
        spec=spec,
        output_dir=out_dir,
        cleanup=bool(args.cleanup),
        label_scheme=args.label_scheme,
        save_pre=bool(args.save_preprocessed),
    )
    print(f"Inference finished ({result['backend']}) in {result['inference_seconds']}s")


if __name__ == "__main__":
    main()
