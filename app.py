from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

# ============================================================
# Tidy Text Suite — Secure Streamlit Demo Shell
# ------------------------------------------------------------
# Purpose:
# A minimal web demo for showcasing functionality without
# exposing core marking/OCR logic in the browser.
#
# Notes:
# - Keep all OCR/AI/marking logic on the server side.
# - Store API keys in Streamlit secrets or environment variables.
# - Do not place prompts, keys, or proprietary logic in frontend JS.
# - Replace the placeholder engine functions below with your
#   existing Tidy Text engine wrappers.
# ============================================================

APP_NAME = "Tidy Text Suite"
APP_VERSION = "0.1.0-demo"
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
# Replace these with wrappers around your existing engine files.
# -----------------------------
def run_conversion(pdf_bytes: bytes, original_name: str) -> ConversionResult:
    """
    Replace with your real conversion pipeline.
    Expected job:
    - Save uploaded PDF temporarily
    - Call OCR / AI transcription engine
    - Return typed text + printable text + report text
    """
    placeholder = (
        f"[DEMO PLACEHOLDER]\n\n"
        f"Original file: {original_name}\n"
        f"Bytes received: {len(pdf_bytes)}\n\n"
        f"This is where the OCR/transcription engine output will appear."
    )
    return ConversionResult(
        typed_text=placeholder,
        printable_text=placeholder,
        report_text="=== TIDY TEXT CONVERSION REPORT ===\n\n" + placeholder,
    )


def run_notes_compare(student_text: str, notes_text: str) -> CompareResult:
    """
    Replace with your real notes comparison / copy detection pipeline.
    """
    result = (
        "=== NOTES COMPARISON REPORT ===\n\n"
        "[DEMO PLACEHOLDER]\n\n"
        f"Student text length: {len(student_text)} characters\n"
        f"Notes text length: {len(notes_text)} characters\n"
    )
    return CompareResult(report_text=result)


def run_marking(student_text: str, criteria_text: str, year_level: str) -> MarkResult:
    """
    Replace with your real rubric-based marking pipeline.
    """
    result = (
        "=== MARKING REPORT ===\n\n"
        "[DEMO PLACEHOLDER]\n\n"
        f"Year level: {year_level}\n"
        f"Student text length: {len(student_text)} characters\n"
        f"Criteria text length: {len(criteria_text)} characters\n"
    )
    return MarkResult(report_text=result)


def run_feedback(student_text: str, criteria_text: str, year_level: str) -> FeedbackResult:
    """
    Replace with your real feedback-generation pipeline.
    """
    result = (
        "=== FEEDBACK REPORT ===\n\n"
        "[DEMO PLACEHOLDER]\n\n"
        f"Year level: {year_level}\n"
        f"Student text length: {len(student_text)} characters\n"
        f"Criteria text length: {len(criteria_text)} characters\n"
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
# Header
# -----------------------------
st.title(f"{APP_NAME} — Web Demo")
st.caption(f"v{APP_VERSION} • {APP_TAGLINE}")

st.info(
    "This demo keeps processing on the server side. Uploaded files and generated outputs are used only for the current session unless you choose to add persistent storage later."
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Workflow")
    st.markdown(
        "1. Upload handwritten PDF\n"
        "2. Convert to typed text\n"
        "3. Optionally compare with notes\n"
        "4. Optionally mark and generate feedback\n"
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
    st.subheader("1. Upload handwritten PDF")
    handwritten_pdf = st.file_uploader(
        "Student handwritten PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a scanned or handwritten student response PDF.",
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
        if handwritten_pdf is None:
            st.error("Please upload a handwritten PDF first.")
        else:
            with st.spinner("Converting handwritten PDF to typed text..."):
                pdf_bytes = save_upload_to_bytes(handwritten_pdf)
                result = run_conversion(pdf_bytes, handwritten_pdf.name)
                st.session_state.converted_text = result.typed_text
                st.session_state.printable_text = result.printable_text
                st.session_state.conversion_report = result.report_text
            st.success("Conversion complete.")

    if compare_notes:
        if not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to compare.")
        elif not notes_text:
            st.error("Please upload or paste study notes text.")
        else:
            with st.spinner("Comparing student response with notes..."):
                result = run_notes_compare(st.session_state.converted_text, notes_text)
                st.session_state.compare_report = result.report_text
            st.success("Notes comparison complete.")

    if mark_work:
        if not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to mark.")
        elif not criteria_text:
            st.error("Please upload or paste criteria / rubric text.")
        else:
            with st.spinner("Marking response..."):
                result = run_marking(st.session_state.converted_text, criteria_text, year_level)
                st.session_state.mark_report = result.report_text
            st.success("Marking complete.")

    if make_feedback:
        if not st.session_state.converted_text:
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
    "Demo shell for Tidy Text Suite. Keep the proprietary OCR, marking, and comparison logic in server-side engine wrappers."
)
