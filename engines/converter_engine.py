from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import re
from pathlib import Path
from typing import Callable, Optional
import threading

import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from openai import OpenAI


ProgressCallback = Optional[Callable[[str], None]]

_DEFAULT_RENDER_DPI = 300


def load_pdf_pages_as_images(pdf_path: str, dpi: int = _DEFAULT_RENDER_DPI) -> list[Image.Image]:
    """Render each PDF page to a PIL image at ``dpi`` (default 300). Requires Poppler (e.g. brew install poppler on macOS)."""
    return convert_from_path(pdf_path, dpi=dpi)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


def get_unique_output_path(path: Path) -> Path:
    """Return ``path`` if it does not exist; otherwise ``name (1).ext``, ``name (2).ext``, …"""
    path = Path(path)
    if not path.exists():
        return path
    parent = path.parent
    suffix = path.suffix
    stem = path.stem
    m = re.fullmatch(r"(.+) \((\d+)\)", stem)
    base_stem = m.group(1) if m else stem
    n = 1
    while True:
        candidate = parent / f"{base_stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


@dataclass
class ConverterJob:
    pdf_path: str
    output_folder: str
    output_name: str
    doc_type: str               # "Exam" or "Study notes"
    engine: str                 # "OpenAI Vision" or "Local Tesseract"
    model_name: str
    out_ext: str                # ".txt" or ".md"
    api_key: Optional[str] = None


@dataclass
class ConverterResult:
    success: bool
    output_path: Optional[str] = None
    full_text: str = ""
    error: Optional[str] = None


def run_converter_job(
    job: ConverterJob,
    progress_callback: ProgressCallback = None,
    cancel_event: Optional[threading.Event] = None,
) -> ConverterResult:
    try:
        _progress(progress_callback, "Starting conversion...\n")
        _check_cancel(cancel_event)

        pdf_path = clean_path(job.pdf_path)
        output_folder = clean_path(job.output_folder)
        output_name = (job.output_name or "").strip()
        out_ext = (job.out_ext or ".txt").strip()

        if not output_name:
            output_name = build_auto_output_name(pdf_path, job.doc_type, out_ext)

        if not output_name.lower().endswith(out_ext.lower()):
            output_name += out_ext

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = get_unique_output_path(output_dir / output_name)

        _progress(progress_callback, f"INPUT : {pdf_path}\n")
        _progress(progress_callback, f"OUTPUT: {output_path}\n")
        _progress(progress_callback, f"TYPE  : {job.doc_type}\n")
        _progress(progress_callback, f"ENGINE: {job.engine}\n")

        client = None
        if job.engine == "OpenAI Vision":
            if not job.api_key:
                raise ValueError("OpenAI API key is required for OpenAI Vision.")
            _progress(progress_callback, f"MODEL : {job.model_name}\n")
            client = OpenAI(api_key=job.api_key)

        _progress(progress_callback, "-" * 60 + "\n\n")
        _check_cancel(cancel_event)

        try:
            page_images = load_pdf_pages_as_images(pdf_path)
        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "Poppler is required to render PDF pages (pdf2image). "
                "On macOS: brew install poppler — then restart the terminal so "
                "`pdftoppm` is on your PATH."
            ) from e

        full_text = build_header(
            source_name=Path(pdf_path).name,
            doc_type=job.doc_type,
            engine=job.engine,
            model_name=job.model_name,
        )

        total = len(page_images)
        for i, img in enumerate(page_images, start=1):
            _check_cancel(cancel_event)
            _progress(progress_callback, f"Processing page {i}/{total}...\n")

            img = _ensure_rgb(img)

            if job.engine == "Local Tesseract":
                text = transcribe_page_with_tesseract(img, cancel_event)
            else:
                text = transcribe_page_with_openai(
                    img=img,
                    client=client,
                    model_name=job.model_name,
                    cancel_event=cancel_event,
                )

            page_block = f"\n\n--- Page {i} ---\n\n{text}"
            full_text += page_block
            _progress(progress_callback, page_block + "\n")

        output_path.write_text(full_text, encoding="utf-8")

        _progress(progress_callback, "\nDone.\n")
        _progress(progress_callback, f"Saved to: {output_path}\n")
        _progress(progress_callback, f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return ConverterResult(
            success=True,
            output_path=str(output_path),
            full_text=full_text,
        )

    except Exception as e:
        return ConverterResult(success=False, error=str(e))


def clean_path(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    path = path.replace("\\ ", " ")
    path = path.replace("\\(", "(").replace("\\)", ")")
    path = path.replace("\\[", "[").replace("\\]", "]")
    path = path.replace("\\&", "&")
    return path


def build_auto_output_name(pdf_path: str, doc_type: str, out_ext: str) -> str:
    base_name = Path(pdf_path).stem
    datestamp = datetime.now().strftime("%Y_%m_%d")
    suffix = "TT_Converted_Exam_Notes" if doc_type == "Study notes" else "TT_Converted_Exam"
    return f"{base_name}_{suffix}_{datestamp}{out_ext}"


def build_header(source_name: str, doc_type: str, engine: str, model_name: str) -> str:
    header_lines = [
        f"ORIGINAL FILE: {source_name}",
        f"DOCUMENT TYPE: {doc_type}",
        f"CONVERTED ON: {datetime.now().strftime('%Y_%m_%d')}",
    ]

    if engine == "OpenAI Vision":
        header_lines.append(f"MODEL USED: {model_name}")
    else:
        header_lines.append("MODEL USED: Local Tesseract")

    return "\n".join(header_lines) + "\n"


def image_to_data_url(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def transcribe_page_with_tesseract(img: Image.Image, cancel_event=None) -> str:
    _check_cancel(cancel_event)
    return pytesseract.image_to_string(img)


def transcribe_page_with_openai(
    img: Image.Image,
    client: OpenAI,
    model_name: str,
    cancel_event=None,
) -> str:
    _check_cancel(cancel_event)

    data_url = image_to_data_url(img)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Carefully transcribe all handwritten and printed text on this exam page. "
                            "Return plain text only. Do not summarize. "
                            "If a word is unclear, write [unclear]."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content or ""


def _progress(callback: ProgressCallback, message: str):
    if callback:
        callback(message)


def _check_cancel(cancel_event: Optional[threading.Event]):
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Run cancelled by user.")