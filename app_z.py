from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from engines.converter_engine import ConverterJob, run_converter_job

# ============================================================
# Tidy Text Suite — Secure Streamlit Demo Shell
# ============================================================

APP_NAME = "Tidy Text Suite"
APP_VERSION = "0.2.0-demo"
APP_TAGLINE = "Handwriting conversion, notes comparison, marking, and feedback"


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class ConversionResult:
    typed_text: str
    printable_text: str
    report_text: str


@dataclass
class CompareResult:
    report_text: str


@dataclass
class MarkResult:
    report_text: str


@dataclass
class FeedbackResult:
    report_text: str


# -----------------------------
# Engine integration points
# -----------------------------
def run_conversion(pdf_bytes: bytes, original_name: str, conversion_mode: str) -> ConversionResult:
    """
    Runs the real converter engine using local Tesseract first.

    conversion_mode values:
    - "Handwritten student response"
    - "Scanned or printed text"
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    output_folder = tempfile.gettempdir()

    job = ConverterJob(
        pdf_path=tmp_path,
        output_folder=output_folder,
        output_name="streamlit_output",
        doc_type="Exam",
        engine="Local Tesseract",
        model_name="",
        out_ext=".txt",
        api_key=None,
    )

    result = run_converter_job(job)
    if not result.success:
        raise RuntimeError(result.error)

    preface_lines = [
        "=== TIDY TEXT CONVERSION REPORT ===",
        "",
        f"Original file: {original_name}",
        f"Conversion mode: {conversion_mode}",
        "Engine used: Local Tesseract",
        "",
    ]

    if conversion_mode == "Scanned or printed text":
        preface_lines.extend(
            [
                "Note:",
                "This option is intended for scanned documents that mainly contain printed text.",
                "If the output quality is poor, try the ChatGPT/OpenAI vision workflow later instead.",
                "",
            ]
        )
    else:
        preface_lines.extend(
            [
                "Note:",
                "This demo is using Local Tesseract for first-pass conversion.",
                "If handwriting accuracy is poor, use the ChatGPT/OpenAI vision workflow for better results.",
                "",
            ]
        )

    report_text = "".join(preface_lines) + result.full_text

    return ConversionResult(
        typed_text=result.full_text,
        printable_text=result.full_text,
        report_text=report_text,
    )


def run_notes_compare(student_text: str, notes_text: str) -> CompareResult:
    result = (
        "=== NOTES COMPARISON REPORT ==="
        "[DEMO PLACEHOLDER]"
        f"Student text length: {len(student_text)} characters"
        f"Notes text length: {len(notes_text)} characters"
    )
    return CompareResult(report_text=result)


def run_marking(student_text: str, criteria_text: str, year_level: str) -> MarkResult:
    result = (
        "=== MARKING REPORT ==="
        "[DEMO PLACEHOLDER]"
        f"Year level: {year_level}"
        f"Student text length: {len(student_text)} characters"
        f"Criteria text length: {len(criteria_text)} characters"
    )
    return MarkResult(report_text=result)


def run_feedback(student_text: str, criteria_text: str, year_level: str) -> FeedbackResult:
    result = (
        "=== FEEDBACK REPORT ==="
        "[DEMO PLACEHOLDER]"
        f"Year level: {year_level}"
        f"Student text length: {len(student_text)} characters"
        f"Criteria text length: {len(criteria_text)} characters"
    )
    return FeedbackResult(report_text=result)


# -----------------------------
# Helpers
# -----------------------------
def download_text_button(label: str, text: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )


def save_upload_to_bytes(uploaded_file) -> bytes:
    return uploaded_file.getvalue() if uploaded_file is not None else b""


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title=f"{APP_NAME} Demo",
    page_icon="📝",
    layout="wide",
)


# -----------------------------
# Basic session state
# -----------------------------
if "converted_text" not in st.session_state:
    st.session_state.converted_text = ""
if "printable_text" not in st.session_state:
    st.session_state.printable_text = ""
if "conversion_report" not in st.session_state:
    st.session_state.conversion_report = ""
if "compare_report" not in st.session_state:
    st.session_state.compare_report = ""
if "mark_report" not in st.session_state:
    st.session_state.mark_report = ""
if "feedback_report" not in st.session_state:
    st.session_state.feedback_report = ""


# -----------------------------
# Usage logging (basic, non-content)
# -----------------------------
import datetime

def log_usage(action: str):
    """Basic usage logger (no content stored)."""
    try:
        import requests
        ip = requests.get("https://api.ipify.org").text
    except Exception:
        ip = "unknown"

    log_line = f"{datetime.datetime.now().isoformat()} | {ip} | {action}"

    with open("usage_log.txt", "a") as f:
        f.write(log_line)


# -----------------------------
# Header
# -----------------------------
st.title(f"{APP_NAME} — Web Demo")
st.caption(f"v{APP_VERSION} • {APP_TAGLINE}")

st.warning(
    "Privacy warning: Do not upload PDFs that contain private or identifying student information. Best practice is to remove, redact, or exclude names, student numbers, addresses, date of birth, school IDs, or any other identifying details before upload."
)

with st.expander("Important use conditions, privacy notice, and disclaimers", expanded=False):
    st.markdown(
        """
**Important privacy and use notice**

- This demo is provided for evaluation and workflow testing only.
- Do not use this app with PDFs that contain personal information that could identify a student or other individual.
- Before upload, remove, redact, or exclude names, student numbers, addresses, dates of birth, school identifiers, signatures, email addresses, phone numbers, and any other identifying details.
- The user is responsible for checking that all uploaded content is appropriate, lawful, and de-identified before use.
- Outputs may contain transcription errors, OCR errors, formatting issues, incorrect marking, incorrect feedback, or incomplete text. All results must be checked by a teacher or other responsible human reviewer before being relied on.
- This tool is an assistive workflow tool only. It does not replace professional judgement, school procedures, moderation, reporting requirements, privacy obligations, or records management obligations.
- No warranty is given that the app will be uninterrupted, error-free, accurate, fit for a particular purpose, or suitable for any compliance obligation.
- By using this demo, the user accepts responsibility for reviewing outputs and for ensuring their own compliance with applicable school, employer, legal, and privacy requirements.
        """
    )

privacy_confirmed = st.checkbox(
    "I confirm that any uploaded PDF has been checked and does not contain private or identifying student information, and does not include unlawful, offensive, or inappropriate material (including hate speech, abuse, or illegal content).",
    value=False,
)

# Copyright notice
st.caption("© Sandle Software — Tidy Text Suite. All rights reserved. This software, including all underlying logic, workflows, and outputs, is the intellectual property of Sandle Software and may not be copied, reproduced, reverse engineered, or redistributed without permission.")
privacy_confirmed = st.checkbox(
    "I confirm that any uploaded PDF has been checked and does not contain private or identifying student information, or that such information has been removed or redacted.",
    value=False,
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Workflow")
    st.markdown(
        "1. Upload handwritten or scanned PDF"
        "2. Convert to typed text"
        "3. Optionally compare with notes"
        "4. Optionally mark and generate feedback"
        "5. Download reports as text"
    )

    year_level = st.selectbox(
        "Year level",
        [
            "Default",
            "Kindergarten",
            "Year 1",
            "Year 2",
            "Year 3",
            "Year 4",
            "Year 5",
            "Year 6",
            "Year 7",
            "Year 8",
            "Year 9",
            "Year 10",
            "Year 11",
            "Year 12",
        ],
        index=0,
    )

    st.subheader("Security notes")
    st.caption(
        "Keep API keys in Streamlit secrets or environment variables. Do not hardcode them into this file."
    )


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("1. Upload source PDF")
    handwritten_pdf = st.file_uploader(
        "Student handwritten or scanned PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a scanned handwritten response or a scanned document with printed text.",
    )

    conversion_mode = st.radio(
        "Conversion type",
        ["Handwritten student response", "Scanned or printed text"],
        index=0,
        help="Choose the option that best matches the document you are uploading.",
    )

    if conversion_mode == "Scanned or printed text":
        st.info(
            "This mode is intended for scanned documents that mainly contain printed text and uses Local Tesseract first. If the result is poor, try the ChatGPT/OpenAI AI model workflow instead."
        )
    else:
        st.info(
            "This demo currently uses Local Tesseract first. For difficult handwriting, the ChatGPT/OpenAI AI model workflow will usually give better results once that path is enabled."
        )

    st.subheader("2. Optional inputs")
    notes_file = st.file_uploader(
        "Study notes or source text (TXT/MD)",
        type=["txt", "md"],
        accept_multiple_files=False,
    )
    criteria_file = st.file_uploader(
        "Marking criteria / rubric (TXT/MD)",
        type=["txt", "md"],
        accept_multiple_files=False,
    )

    notes_text_manual = st.text_area(
        "Or paste notes text",
        height=140,
        placeholder="Paste study notes here if not uploading a text file.",
    )
    criteria_text_manual = st.text_area(
        "Or paste criteria / rubric text",
        height=140,
        placeholder="Paste criteria or rubric here if not uploading a text file.",
    )

    st.subheader("3. Actions")
    convert_only = st.button("Convert PDF to text", use_container_width=True, type="primary")
    compare_notes = st.button("Compare with notes", use_container_width=True)
    mark_work = st.button("Mark response", use_container_width=True)
    make_feedback = st.button("Generate feedback", use_container_width=True)

    notes_text = ""
    if notes_file is not None:
        notes_text = notes_file.getvalue().decode("utf-8", errors="replace")
    elif notes_text_manual.strip():
        notes_text = notes_text_manual.strip()

    criteria_text = ""
    if criteria_file is not None:
        criteria_text = criteria_file.getvalue().decode("utf-8", errors="replace")
    elif criteria_text_manual.strip():
        criteria_text = criteria_text_manual.strip()

    if convert_only:
        log_usage("convert")
        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before uploading or processing any PDF.")
        elif handwritten_pdf is None:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Converting PDF to typed text..."):
                pdf_bytes = save_upload_to_bytes(handwritten_pdf)
                result = run_conversion(pdf_bytes, handwritten_pdf.name, conversion_mode)
                st.session_state.converted_text = result.typed_text
                st.session_state.printable_text = result.printable_text
                st.session_state.conversion_report = result.report_text
            st.success("Conversion complete.")

    if compare_notes:
        log_usage("compare")
        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to compare.")
        elif not notes_text:
            st.error("Please upload or paste study notes text.")
        else:
            with st.spinner("Comparing student response with notes..."):
                result = run_notes_compare(st.session_state.converted_text, notes_text)
                st.session_state.compare_report = result.report_text
            st.success("Notes comparison complete.")

    if mark_work:
        log_usage("mark")
        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to mark.")
        elif not criteria_text:
            st.error("Please upload or paste criteria / rubric text.")
        else:
            with st.spinner("Marking response..."):
                result = run_marking(st.session_state.converted_text, criteria_text, year_level)
                st.session_state.mark_report = result.report_text
            st.success("Marking complete.")

    if make_feedback:
        log_usage("feedback")
        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text for feedback.")
        elif not criteria_text:
            st.error("Please upload or paste criteria / rubric text.")
        else:
            with st.spinner("Generating feedback..."):
                result = run_feedback(st.session_state.converted_text, criteria_text, year_level)
                st.session_state.feedback_report = result.report_text
            st.success("Feedback complete.")

with right_col:
    st.subheader("Outputs")

    output_tabs = st.tabs(
        [
            "Typed text",
            "Printable text",
            "Conversion report",
            "Notes compare",
            "Mark report",
            "Feedback report",
        ]
    )

    with output_tabs[0]:
        st.text_area(
            "Typed text output",
            value=st.session_state.converted_text,
            height=260,
            key="typed_text_area",
        )
        if st.session_state.converted_text:
            download_text_button(
                "Download typed text",
                st.session_state.converted_text,
                "typed_text.txt",
            )

    with output_tabs[1]:
        st.text_area(
            "Printable text output",
            value=st.session_state.printable_text,
            height=260,
            key="printable_text_area",
        )
        if st.session_state.printable_text:
            download_text_button(
                "Download printable text",
                st.session_state.printable_text,
                "printable_text.txt",
            )

    with output_tabs[2]:
        st.text_area(
            "Conversion report",
            value=st.session_state.conversion_report,
            height=260,
            key="conversion_report_area",
        )
        if st.session_state.conversion_report:
            download_text_button(
                "Download conversion report",
                st.session_state.conversion_report,
                "conversion_report.txt",
            )

    with output_tabs[3]:
        st.text_area(
            "Notes comparison report",
            value=st.session_state.compare_report,
            height=260,
            key="compare_report_area",
        )
        if st.session_state.compare_report:
            download_text_button(
                "Download notes comparison report",
                st.session_state.compare_report,
                "notes_compare_report.txt",
            )

    with output_tabs[4]:
        st.text_area(
            "Marking report",
            value=st.session_state.mark_report,
            height=260,
            key="mark_report_area",
        )
        if st.session_state.mark_report:
            download_text_button(
                "Download marking report",
                st.session_state.mark_report,
                "marking_report.txt",
            )

    with output_tabs[5]:
        st.text_area(
            "Feedback report",
            value=st.session_state.feedback_report,
            height=260,
            key="feedback_report_area",
        )
        if st.session_state.feedback_report:
            download_text_button(
                "Download feedback report",
                st.session_state.feedback_report,
                "feedback_report.txt",
            )


st.divider()
st.caption(
    "Demo shell for Tidy Text Suite. Keep the proprietary OCR, marking, and comparison logic in server-side engine wrappers. Results must always be checked by a human reviewer."
)
