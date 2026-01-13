"""Modal app definition and inference function for SHARP.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import modal

from sharp.modal.image import create_modal_image

if TYPE_CHECKING:
    import numpy as np

    from sharp.utils.gaussians import Gaussians3D

LOGGER = logging.getLogger(__name__)

# Modal app configuration
APP_NAME = "sharp-gaussian-splat"
VOLUME_NAME = "sharp-model-cache"
MODEL_CACHE_PATH = "/cache/models"
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
TIMEOUT_SECONDS = 300

# GPU type mapping
GpuTier = Literal["t4", "l4", "a10", "a100", "h100"]

# Create Modal app and volume
app = modal.App(name=APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
modal_image = create_modal_image()


def _load_image_from_bytes(image_bytes: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Load an image from bytes and extract focal length.

    Args:
        image_bytes: Raw image file bytes.
        filename: Original filename (used for format detection).

    Returns:
        Tuple of (image array, focal length in pixels).
    """
    import numpy as np
    import pillow_heif
    from PIL import ExifTags, Image

    file_ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(image_bytes)

    if file_ext in [".heic"]:
        heif_file = pillow_heif.open_heif(buffer, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(buffer)

    # Extract EXIF data
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    # Handle image orientation
    exif_orientation = exif_dict.get("Orientation", 1)
    if exif_orientation == 3:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_180)
    elif exif_orientation == 6:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_270)
    elif exif_orientation == 8:
        img_pil = img_pil.transpose(Image.Transpose.ROTATE_90)

    # Extract focal length
    f_35mm = exif_dict.get("FocalLengthIn35mmFilm", exif_dict.get("FocalLenIn35mmFilm"))
    if f_35mm is None or f_35mm < 1:
        f_35mm = exif_dict.get("FocalLength")
        if f_35mm is None:
            LOGGER.warning(f"No focal length in EXIF for {filename}, using 30mm default.")
            f_35mm = 30.0
        elif f_35mm < 10.0:
            # Crude approximation for non-35mm sensors
            f_35mm *= 8.4

    img = np.asarray(img_pil)

    # Convert to RGB if needed
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))
    img = img[:, :, :3]  # Remove alpha if present

    # Convert focal length to pixels
    height, width = img.shape[:2]
    f_px = f_35mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)

    return img, f_px


def _serialize_ply_to_bytes(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int]
) -> bytes:
    """Serialize Gaussians3D to PLY bytes.

    Args:
        gaussians: The Gaussians3D to serialize.
        f_px: Focal length in pixels.
        image_shape: Image dimensions as (height, width).

    Returns:
        PLY file content as bytes.
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement

    from sharp.utils import color_space as cs_utils
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    # Convert linearRGB to sRGB for compatibility with public renderers
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (xyz, colors, opacity_logits, scale_logits, quaternions),
        dim=1,
    )

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    image_height, image_width = image_shape

    # Export image size
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # Export intrinsics
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # Export dummy extrinsics
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # Export frame info
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # Export disparity ranges
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # Export colorspace
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    # Export version
    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )

    # Write to bytes
    buffer = io.BytesIO()
    plydata.write(buffer)
    buffer.seek(0)
    return buffer.read()


def _serialize_splat_to_bytes(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int]
) -> bytes:
    """Serialize Gaussians3D to SPLAT bytes."""
    import numpy as np

    from sharp.utils import color_space as cs_utils

    xyz = gaussians.mean_vectors.flatten(0, 1).cpu().numpy()
    scales = gaussians.singular_values.flatten(0, 1).cpu().numpy()
    quats = gaussians.quaternions.flatten(0, 1).cpu().numpy()
    colors_rgb = cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1)).cpu().numpy()
    opacities = gaussians.opacities.flatten(0, 1).cpu().numpy()

    sort_idx = np.argsort(-(scales.prod(axis=1) * opacities))
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    buffer = io.BytesIO()
    for i in sort_idx:
        buffer.write(xyz[i].astype(np.float32).tobytes())
        buffer.write(scales[i].astype(np.float32).tobytes())
        rgba = np.concatenate([colors_rgb[i], [opacities[i]]])
        buffer.write((rgba * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write((quats[i] * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())

    buffer.seek(0)
    return buffer.read()


def _serialize_sog_to_bytes(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int]
) -> bytes:
    """Serialize Gaussians3D to SOG bytes (zip archive)."""
    import io as stdlib_io
    import json
    import math
    import zipfile

    import numpy as np
    from PIL import Image

    from sharp.utils import color_space as cs_utils

    xyz = gaussians.mean_vectors.flatten(0, 1).cpu().numpy()
    scales = gaussians.singular_values.flatten(0, 1).cpu().numpy()
    quats = gaussians.quaternions.flatten(0, 1).cpu().numpy()
    colors_srgb = cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1)).cpu().numpy()
    opacities = gaussians.opacities.flatten(0, 1).cpu().numpy()

    num_gaussians = len(xyz)
    xyz_raw = xyz

    img_width = int(math.ceil(math.sqrt(num_gaussians)))
    img_height = int(math.ceil(num_gaussians / img_width))
    total_pixels = img_width * img_height

    def pad_array(arr: np.ndarray, total: int) -> np.ndarray:
        if len(arr) < total:
            pad_shape = (total - len(arr),) + arr.shape[1:]
            return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)])
        return arr

    xyz = pad_array(xyz, total_pixels)
    scales = pad_array(scales, total_pixels)
    quats = pad_array(quats, total_pixels)
    colors_srgb = pad_array(colors_srgb, total_pixels)
    opacities = pad_array(opacities, total_pixels)

    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

    def symlog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    xyz_log = symlog(xyz)
    mins = xyz_log.min(axis=0)
    maxs = xyz_log.max(axis=0)
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-8, 1.0, ranges)

    xyz_norm = (xyz_log - mins) / ranges
    xyz_q16 = (xyz_norm * 65535).clip(0, 65535).astype(np.uint16)

    means_l = (xyz_q16 & 0xFF).astype(np.uint8)
    means_u = (xyz_q16 >> 8).astype(np.uint8)

    def encode_quaternion(q: np.ndarray) -> np.ndarray:
        abs_q = np.abs(q)
        mode = np.argmax(abs_q, axis=1)
        signs = np.sign(q[np.arange(len(q)), mode])
        q = q * signs[:, None]

        result = np.zeros((len(q), 4), dtype=np.uint8)
        sqrt2_inv = 1.0 / math.sqrt(2)

        for i in range(len(q)):
            m = mode[i]
            kept = [j for j in range(4) if j != m]
            vals = q[i, kept]
            encoded = ((vals * sqrt2_inv + 0.5) * 255).clip(0, 255).astype(np.uint8)
            result[i, :3] = encoded
            result[i, 3] = 252 + m

        return result

    quats_encoded = encode_quaternion(quats)

    scales_log = np.log(np.maximum(scales, 1e-10))
    scales_log_flat = scales_log.flatten()

    percentiles = np.linspace(0, 100, 256)
    scale_codebook = np.percentile(scales_log_flat, percentiles).astype(np.float32)

    def quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(codebook, values)
        indices = np.clip(indices, 0, len(codebook) - 1)
        prev_indices = np.clip(indices - 1, 0, len(codebook) - 1)
        dist_curr = np.abs(values - codebook[indices])
        dist_prev = np.abs(values - codebook[prev_indices])
        use_prev = (dist_prev < dist_curr) & (indices > 0)
        indices = np.where(use_prev, prev_indices, indices)
        return indices.astype(np.uint8)

    scales_q = np.stack(
        [
            quantize_to_codebook(scales_log[:, 0], scale_codebook),
            quantize_to_codebook(scales_log[:, 1], scale_codebook),
            quantize_to_codebook(scales_log[:, 2], scale_codebook),
        ],
        axis=1,
    )

    SH_C0 = 0.28209479177387814
    sh0_coeffs = (colors_srgb - 0.5) / SH_C0
    sh0_flat = sh0_coeffs.flatten()

    sh0_percentiles = np.linspace(0, 100, 256)
    sh0_codebook = np.percentile(sh0_flat, sh0_percentiles).astype(np.float32)

    sh0_r = quantize_to_codebook(sh0_coeffs[:, 0], sh0_codebook)
    sh0_g = quantize_to_codebook(sh0_coeffs[:, 1], sh0_codebook)
    sh0_b = quantize_to_codebook(sh0_coeffs[:, 2], sh0_codebook)
    sh0_a = (opacities * 255).clip(0, 255).astype(np.uint8)

    def create_image(data: np.ndarray, width: int, height: int) -> Image.Image:
        data = data.reshape(height, width, -1)
        if data.shape[2] == 3:
            return Image.fromarray(data, mode="RGB")
        if data.shape[2] == 4:
            return Image.fromarray(data, mode="RGBA")
        raise ValueError(f"Unexpected channel count: {data.shape[2]}")

    means_l_img = create_image(means_l, img_width, img_height)
    means_u_img = create_image(means_u, img_width, img_height)
    quats_img = create_image(quats_encoded, img_width, img_height)
    scales_img = create_image(scales_q, img_width, img_height)

    sh0_data = np.stack([sh0_r, sh0_g, sh0_b, sh0_a], axis=1)
    sh0_img = create_image(sh0_data, img_width, img_height)

    image_height, image_width = image_shape
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ],
        dtype=np.float32,
    )
    extrinsic = np.eye(4, dtype=np.float32)
    frame = np.array([1, num_gaussians], dtype=np.int32)
    disparity = 1.0 / xyz_raw[:, 2]
    disparity_quantiles = np.quantile(disparity, [0.1, 0.9]).astype(np.float32)
    color_space_index = cs_utils.encode_color_space("sRGB")
    ply_version = np.array([1, 5, 0], dtype=np.uint8)

    meta = {
        "version": 2,
        "count": num_gaussians,
        "antialias": False,
        "means": {
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {"codebook": scale_codebook.tolist(), "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": sh0_codebook.tolist(), "files": ["sh0.webp"]},
        "sharp_metadata": {
            "image_size": [int(image_width), int(image_height)],
            "intrinsic": intrinsic.flatten().tolist(),
            "extrinsic": extrinsic.flatten().tolist(),
            "frame": frame.tolist(),
            "disparity": disparity_quantiles.tolist(),
            "color_space": int(color_space_index),
            "version": ply_version.tolist(),
        },
    }

    buffer = stdlib_io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        for name, img in [
            ("means_l.webp", means_l_img),
            ("means_u.webp", means_u_img),
            ("quats.webp", quats_img),
            ("scales.webp", scales_img),
            ("sh0.webp", sh0_img),
        ]:
            img_buf = stdlib_io.BytesIO()
            img.save(img_buf, format="WEBP", lossless=True)
            zf.writestr(name, img_buf.getvalue())

        zf.writestr("meta.json", json.dumps(meta, indent=2))

    buffer.seek(0)
    return buffer.read()


def _serialize_outputs(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    filename: str,
    export_formats: Sequence[str],
) -> list[tuple[str, bytes]]:
    base = Path(filename).stem
    outputs: list[tuple[str, bytes]] = []

    for fmt in export_formats:
        if fmt == "ply":
            outputs.append((f"{base}.ply", _serialize_ply_to_bytes(gaussians, f_px, image_shape)))
        elif fmt == "splat":
            outputs.append(
                (f"{base}.splat", _serialize_splat_to_bytes(gaussians, f_px, image_shape))
            )
        elif fmt == "sog":
            outputs.append((f"{base}.sog", _serialize_sog_to_bytes(gaussians, f_px, image_shape)))
        else:
            LOGGER.warning("Unknown export format: %s", fmt)

    return outputs


# GPU-specific function variants
@app.function(
    gpu="t4",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_t4(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Run inference on T4 GPU ($0.59/hr, budget option)."""
    return _predict_impl(image_bytes, filename, export_formats)


@app.function(
    gpu="l4",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_l4(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Run inference on L4 GPU ($0.80/hr)."""
    return _predict_impl(image_bytes, filename, export_formats)


@app.function(
    gpu="a10",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_a10(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Run inference on A10 GPU ($1.10/hr, default)."""
    return _predict_impl(image_bytes, filename, export_formats)


@app.function(
    gpu="a100",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_a100(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Run inference on A100 GPU ($2.50/hr)."""
    return _predict_impl(image_bytes, filename, export_formats)


@app.function(
    gpu="h100",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=modal_image,
)
def predict_gaussian_splat_h100(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Run inference on H100 GPU ($3.95/hr, fastest)."""
    return _predict_impl(image_bytes, filename, export_formats)


def _predict_impl(
    image_bytes: bytes, filename: str, export_formats: Sequence[str] | None = None
) -> list[tuple[str, bytes]]:
    """Shared implementation for all GPU variants.

    This is called by the GPU-specific functions and contains the actual
    inference logic (same as predict_gaussian_splat).
    """
    import torch
    import torch.nn.functional as F

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import unproject_gaussians

    LOGGER.info(f"Processing {filename} on Modal GPU")

    # Load image from bytes
    image, f_px = _load_image_from_bytes(image_bytes, filename)
    height, width = image.shape[:2]

    device = torch.device("cuda")

    # Load or download model
    model_path = Path(MODEL_CACHE_PATH) / "sharp_model.pt"

    def download_model() -> dict:
        """Download model from URL and cache to volume."""
        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True, map_location=device
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, model_path)
        model_volume.commit()
        LOGGER.info("Model cached to volume")
        return state

    if model_path.exists():
        LOGGER.info("Loading cached model from volume")
        try:
            state_dict = torch.load(model_path, weights_only=True, map_location=device)
        except Exception as e:
            LOGGER.warning(f"Cached model is corrupted: {e}")
            LOGGER.info("Deleting corrupted cache and re-downloading...")
            model_path.unlink()
            model_volume.commit()
            state_dict = download_model()
    else:
        state_dict = download_model()

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running inference")
    with torch.no_grad():
        gaussians_ndc = gaussian_predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )

    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    export_formats = tuple(fmt.lower() for fmt in (export_formats or ("ply",)))

    LOGGER.info("Serializing outputs: %s", ", ".join(export_formats))
    outputs = _serialize_outputs(gaussians, f_px, (height, width), filename, export_formats)

    LOGGER.info(f"Done processing {filename}")
    return outputs


def get_predict_function(gpu_tier: GpuTier = "a10"):
    """Get the appropriate predict function for the GPU tier.

    Args:
        gpu_tier: One of 't4', 'l4', 'a10', 'a100', 'h100'.

    Returns:
        The Modal function for the specified GPU tier.
    """
    functions = {
        "t4": predict_gaussian_splat_t4,
        "l4": predict_gaussian_splat_l4,
        "a10": predict_gaussian_splat_a10,
        "a100": predict_gaussian_splat_a100,
        "h100": predict_gaussian_splat_h100,
    }
    return functions[gpu_tier]
