"""Modal FastAPI endpoint for SHARP inference and Supabase upload."""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Sequence

import modal
from fastapi import HTTPException, Request, Response

# Secret names expected to be provisioned in Modal
SUPABASE_SECRET_NAME = "supabase-creds"
API_AUTH_SECRET_NAME = "sharp-api-auth"

API_KEY_HEADER = "X-API-KEY"
API_KEY_ENV = "API_AUTH_TOKEN"
SUPABASE_URL_ENV = "SUPABASE_URL"
SUPABASE_KEY_ENV = "SUPABASE_KEY"
SUPABASE_BUCKET_ENV = "SUPABASE_BUCKET"

DEFAULT_BUCKET = "testbucket"
DEFAULT_EXPORT_FORMATS: Sequence[str] = ("sog",)

# Local repo path (resolved at deploy time on your machine, not in container)
REMOTE_REPO_PATH = "/root/ml-sharp"
REMOTE_SRC_PATH = f"{REMOTE_REPO_PATH}/src"

# Only compute REPO_ROOT when running locally (during deploy)
_is_modal_container = os.environ.get("MODAL_IS_REMOTE", "") == "1"
if _is_modal_container:
    REPO_ROOT = None  # Not used in container
else:
    REPO_ROOT = Path(__file__).resolve().parents[3]  # c:\modelSharp\ml-sharp

# Build a Modal image with SHARP + API dependencies
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "timm>=1.0.0",
        "scipy>=1.11.0",
        "plyfile>=1.0.0",
        "imageio>=2.30.0",
        "pillow-heif>=0.16.0",
        "numpy>=1.24.0",
        "click>=8.0.0",
        "fastapi",
        "python-multipart",
        "supabase",
    )
    .run_commands("pip install gsplat --no-build-isolation")
    .env({"PYTHONPATH": REMOTE_SRC_PATH, "MODAL_IS_REMOTE": "1"})
    .add_local_dir(
        str(REPO_ROOT) if REPO_ROOT else "/dummy",
        REMOTE_REPO_PATH,
        ignore=[".git", ".venv", "__pycache__", "node_modules", ".conda", "venv"],
    )
)

# Import volume/constants from app.py at deploy time (runs locally)
from sharp.modal.app import MODEL_CACHE_PATH, TIMEOUT_SECONDS, model_volume

# Separate Modal app for the HTTP endpoint
app = modal.App(name="sharp-api")


@app.function(
    gpu="a10",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=api_image,
    secrets=[
        modal.Secret.from_name(SUPABASE_SECRET_NAME),
        modal.Secret.from_name(API_AUTH_SECRET_NAME),
    ],
)
@modal.fastapi_endpoint(method="POST")
async def process_image(request: Request):
    """Run SHARP on an uploaded image, upload outputs to Supabase, and return URLs."""
    # Import inside function so it runs in the container with PYTHONPATH set
    from sharp.modal.app import _predict_impl
    from supabase import create_client

    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        if request.headers.get(API_KEY_HEADER) != api_key:
            return Response(status_code=401)

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        raise HTTPException(status_code=400, detail="Form field 'file' is required.")

    filename = upload.filename or "upload.png"
    image_bytes = await upload.read()

    formats_raw = form.get("format") or form.get("formats")
    export_formats: Sequence[str]
    if isinstance(formats_raw, str) and formats_raw.strip():
        export_formats = [fmt.strip() for fmt in formats_raw.split(",") if fmt.strip()]
    else:
        export_formats = DEFAULT_EXPORT_FORMATS

    outputs = _predict_impl(image_bytes=image_bytes, filename=filename, export_formats=export_formats)

    supabase_url = os.environ.get(SUPABASE_URL_ENV)
    supabase_key = os.environ.get(SUPABASE_KEY_ENV)
    bucket = os.environ.get(SUPABASE_BUCKET_ENV, DEFAULT_BUCKET)

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured.")

    client = create_client(supabase_url, supabase_key)
    prefix = form.get("prefix") or "collections/default/assets"

    uploaded_files: list[dict[str, str]] = []
    for output_name, file_bytes in outputs:
        object_key = str(Path(prefix) / output_name)
        file_handle = BytesIO(file_bytes)

        client.storage.from_(bucket).upload(
            object_key,
            file_handle.getvalue(),
            {
                "content-type": "application/octet-stream",
                "upsert": "true",  # must be str, not bool
            },
        )

        url = client.storage.from_(bucket).get_public_url(object_key)
        uploaded_files.append({"name": output_name, "path": object_key, "url": url})

    return {"files": uploaded_files}