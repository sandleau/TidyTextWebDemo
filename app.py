from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from openai import OpenAI

import tidy_text_v2 as backend
from engines.converter_engine import ConverterJob, run_converter_job


APP_NAME = "Tidy Text Suite"
APP_VERSION = "0.4.0-demo"
APP_TAGLINE = "Handwriting conversion, notes comparison, marking, and feedback"

# Change these if your desktop build uses different working models
DEFAULT_VISION_MODEL = "gpt-5.4-mini"
DEFAULT_TEXT_MODEL = "gpt-5.4-mini"


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


def get_api_key() -> tuple[Optional[str], str]:
    """
    Priority:
    1. Session key entered by user in sidebar
    2. Streamlit secrets
    3. Environment variable
    """
    session_key = st.session_state.get("session_api_key", "").strip()
    if session_key:
        return session_key, "session key"

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "").strip()
        if secret_key:
            return secret_key, "streamlit secrets"
    except Exception:
        pass

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "environment variable"

    return None, "not set"


def log_usage(action: str, engine_used: str = "") -> None:
    """
    Basic usage logger.
    No document content is stored.
    """
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text
    except Exception:
        ip = "unknown"

    timestamp = datetime.now().isoformat()
    key_source = st.session_state.get("api_key_source_label", "unknown")
    line = f"{timestamp} | {ip} | {action} | {engine_used} | key_source={key_source}\n"

    with open("usage_log.txt", "a", encoding="utf-8") as f:
        f.write(line)


def extract_copy_band(ai_text: str, fallback: str) -> str:
    match = re.search(r"Similarity:\s*(LOW|MEDIUM|MEDIUM-HIGH|HIGH)", ai_text, re.I)
    return match.group(1).upper() if match else fallback


# -----------------------------
# Engine-backed functions
# -----------------------------
def run_conversion(
    pdf_bytes: bytes,
    original_name: str,
    conversion_mode: str,
    api_key: Optional[str],
    vision_model: str,
) -> ConversionResult:
    """
    Printed/scanned text -> Local Tesseract
    Handwritten response  -> OpenAI Vision
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    output_folder = tempfile.gettempdir()

    if conversion_mode == "Handwritten student response":
        engine_name = "OpenAI Vision"
        model_name = vision_model
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key is available for handwriting OCR. "
                "Add one in the sidebar for this session, or configure OPENAI_API_KEY."
            )
    else:
        engine_name = "Local Tesseract"
        model_name = ""
        api_key = None

    job = ConverterJob(
        pdf_path=tmp_path,
        output_folder=output_folder,
        output_name="streamlit_output",
        doc_type="Exam",
        engine=engine_name,
        model_name=model_name,
        out_ext=".txt",
        api_key=api_key,
    )

    result = run_converter_job(job)
    if not result.success:
        raise RuntimeError(result.error)

    preface_lines = [
        "=== TIDY TEXT CONVERSION REPORT ===",
        "",
        f"Original file: {original_name}",
        f"Conversion mode: {conversion_mode}",
        f"Engine used: {engine_name}",
    ]

    if engine_name == "OpenAI Vision":
        preface_lines.append(f"Model used: {model_name}")

    preface_lines.extend(
        [
            "",
            "Note:",
            "All OCR output must be reviewed by a human before use.",
            "",
        ]
    )

    report_text = "\n".join(preface_lines) + result.full_text

    return ConversionResult(
        typed_text=result.full_text,
        printable_text=result.full_text,
        report_text=report_text,
    )


def run_notes_compare(student_text: str, notes_text: str, api_key: Optional[str]) -> CompareResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for notes comparison. "
            "Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    backend.client = OpenAI(api_key=api_key)
    copy_result = backend.check_copying(notes_text, student_text)

    ai_text = copy_result["ai"]
    band = extract_copy_band(ai_text, copy_result.get("suggested_band", "LOW"))

    report_lines = [
        "=== NOTES COMPARISON REPORT ===",
        "",
        f"Copy Check Band: {band}",
        "",
        f"Phrase Overlap: {copy_result.get('phrase_overlap', 'N/A')}%",
        f"Sentence Similarity: {copy_result.get('sentence_similarity', 'N/A')}%",
        "",
        "AI Review:",
        ai_text,
    ]

    return CompareResult(report_text="\n".join(report_lines))


def run_marking(
    student_text: str,
    criteria_text: str,
    year_level: str,
    api_key: Optional[str],
    notes_text: Optional[str] = None,
) -> MarkResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for marking. "
            "Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    backend.client = OpenAI(api_key=api_key)

    questions = backend.parse_exam_questions(student_text)
    question_keys = sorted(questions.keys(), key=backend.sort_key)

    if not question_keys:
        raise RuntimeError(
            "No questions were detected in the converted text. "
            "Check the OCR output first or improve the question numbering/format."
        )

    marking_guide = backend.build_marking_guide(criteria_text, question_keys)

    report_lines = [
        "=== TIDY TEXT - WRITING MARKER REPORT ===",
        "",
        f"Year level: {year_level}",
        f"Study notes used: {'Yes' if notes_text else 'No'}",
        "",
        "=== INFERRED MARKING GUIDE ===",
        "",
        json.dumps(marking_guide, indent=2, ensure_ascii=False),
        "",
        "=" * 60,
        "",
        "=== COMBINED ASSESSMENT REPORT ===",
        "",
    ]

    total_awarded = 0
    total_possible = 0
    highest_flag = "LOW"

    for key in question_keys:
        answer = questions[key]

        if key not in marking_guide:
            report_lines.extend(
                [
                    key,
                    "",
                    "No inferred marking guide for this question.",
                    "",
                    "-" * 60,
                    "",
                ]
            )
            continue

        guide = marking_guide[key]
        mark_result = backend.mark_response(
            question=guide["question"],
            criteria=guide["criteria"],
            answer=answer,
            max_mark=guide["max_mark"],
            year_level=year_level,
        )

        max_mark = int(guide["max_mark"])
        total_possible += max_mark

        match = re.search(rf"Mark:\s*(\d+)\s*/\s*{max_mark}", mark_result)
        if match:
            total_awarded += int(match.group(1))

        report_lines.extend(
            [
                key,
                "",
                mark_result,
                "",
            ]
        )

        if notes_text:
            copy_result = backend.check_copying(notes_text, answer)
            ai_text = copy_result["ai"]
            band = extract_copy_band(ai_text, copy_result.get("suggested_band", "LOW"))

            if backend.flag_rank(band) > backend.flag_rank(highest_flag):
                highest_flag = band

            report_lines.extend(
                [
                    f"Copy Check Band: {band}",
                    "",
                    f"Phrase Overlap: {copy_result.get('phrase_overlap', 'N/A')}%",
                    f"Sentence Similarity: {copy_result.get('sentence_similarity', 'N/A')}%",
                    "",
                    "Copy Check Review:",
                    ai_text,
                    "",
                ]
            )

        report_lines.extend(["-" * 60, ""])

    report_lines.append(f"TOTAL: {total_awarded}/{total_possible}")

    if notes_text:
        report_lines.extend(["", f"OVERALL COPY FLAG: {highest_flag}"])

    return MarkResult(report_text="\n".join(report_lines))


def run_feedback(
    student_text: str,
    criteria_text: str,
    year_level: str,
    api_key: Optional[str],
    text_model: str,
) -> FeedbackResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for feedback generation. "
            "Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are helping a teacher write feedback on student work.

Year level: {year_level}

Please read the student's response and the marking criteria.
Write clear, teacher-friendly feedback in plain text.

Requirements:
- Do not mention private data.
- Keep it concise and practical.
- Include these headings:
  Strengths
  Next steps
  Suggested improvement actions
- Avoid giving legal, medical, or personal advice.
- Base the feedback only on the provided text.

STUDENT RESPONSE:
{student_text}

MARKING CRITERIA:
{criteria_text}
""".strip()

    response = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": "You write clear, concise educational feedback in plain text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = response.choices[0].message.content or ""

    report_lines = [
        "=== FEEDBACK REPORT ===",
        "",
        f"Year level: {year_level}",
        f"Model used: {text_model}",
        "",
        text,
    ]

    return FeedbackResult(report_text="\n".join(report_lines))


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title=f"{APP_NAME} Demo",
    page_icon="📝",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "converted_text": "",
    "printable_text": "",
    "conversion_report": "",
    "compare_report": "",
    "mark_report": "",
    "feedback_report": "",
    "session_api_key": "",
    "api_key_source_label": "not set",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Workflow")
    st.markdown(
        "**Printed / scanned text**\n"
        "Upload PDF → Convert with traditional OCR → Review → Download or continue to compare/mark\n\n"
        "**Handwritten student work**\n"
        "Upload PDF → Convert with AI OCR → Review → Continue to compare, mark, or feedback"
    )

    st.subheader("OpenAI session settings")
    st.caption(
        "Handwritten OCR, notes comparison, marking, and feedback use OpenAI. "
        "Printed/scanned OCR uses Local Tesseract."
    )

    st.text_input(
        "Session API key override",
        type="password",
        key="session_api_key",
        help="Optional. Paste your own OpenAI API key for this browser session if the default key is unavailable or out of credit.",
    )

    api_key, api_key_source = get_api_key()
    st.session_state.api_key_source_label = api_key_source

    if api_key:
        st.success(f"API key available via {api_key_source}.")
    else:
        st.warning("No OpenAI API key detected yet.")

    with st.expander("How to use your own API key for this session", expanded=False):
        st.markdown(
            """
1. Paste your OpenAI API key into **Session API key override** above.  
2. It will be used only for this current Streamlit session.  
3. Handwritten OCR, notes comparison, marking, and feedback will then use that key.  
4. Printed/scanned OCR does not need an API key because it uses Local Tesseract.

This app does not intentionally save the session key into the document outputs.
"""
        )

    vision_model = st.text_input(
        "AI OCR model",
        value=DEFAULT_VISION_MODEL,
        help="Used for handwriting OCR.",
    )

    text_model = st.text_input(
        "AI text model",
        value=DEFAULT_TEXT_MODEL,
        help="Used for notes comparison, marking, and feedback.",
    )

    st.subheader("Year level")
    year_level = st.selectbox(
        "Select year level",
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
        label_visibility="collapsed",
    )

    st.subheader("Security notes")
    st.caption(
        "Keep long-term API keys in Streamlit secrets or environment variables. "
        "Do not hardcode them into this file."
    )


# -----------------------------
# Header
# -----------------------------
st.title(f"{APP_NAME} — Web Demo")
st.caption(f"v{APP_VERSION} • {APP_TAGLINE}")

st.warning(
    "Privacy warning: Do not upload PDFs that contain private or identifying student information. "
    "Best practice is to remove, redact, or exclude names, student numbers, addresses, date of birth, "
    "school IDs, or any other identifying details before upload."
)

st.info(
    "Recommended workflow: Printed or scanned text should use traditional OCR (Tesseract). "
    "Handwritten student responses should use AI OCR."
)

with st.expander("Important use conditions, privacy notice, and disclaimers", expanded=False):
    st.markdown(
        """
**Important privacy and use notice**

- This demo is provided for evaluation and workflow testing only.
- Do not use this app with PDFs that contain personal information that could identify a student or other individual.
- Before upload, remove, redact, or exclude names, student numbers, addresses, dates of birth, school identifiers, signatures, email addresses, phone numbers, and any other identifying details.
- The user is responsible for checking that all uploaded content is appropriate, lawful, and de-identified before use.
- Do not upload unlawful, abusive, hateful, offensive, or otherwise inappropriate material.
- Outputs may contain transcription errors, OCR errors, formatting issues, incorrect marking, incorrect feedback, or incomplete text. All results must be checked by a teacher or other responsible human reviewer before being relied on.
- This tool is an assistive workflow tool only. It does not replace professional judgement, school procedures, moderation, reporting requirements, privacy obligations, or records management obligations.
- No warranty is given that the app will be uninterrupted, error-free, accurate, fit for a particular purpose, or suitable for any compliance obligation.
- By using this demo, the user accepts responsibility for reviewing outputs and for ensuring their own compliance with applicable school, employer, legal, and privacy requirements.
"""
    )

privacy_confirmed = st.checkbox(
    "I confirm that any uploaded PDF has been checked and does not contain private or identifying student information, "
    "and does not include unlawful, offensive, or inappropriate material (including hate speech, abuse, or illegal content).",
    value=False,
)

st.caption(
    "© Sandle Software — Tidy Text Suite. All rights reserved. "
    "This software, including all underlying logic, workflows, and outputs, is the intellectual property "
    "of Sandle Software and may not be copied, reproduced, reverse engineered, or redistributed without permission."
)


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("1. Upload source PDF")
    st.caption("Choose the conversion path that matches the document type before processing.")

    handwritten_pdf = st.file_uploader(
        "Student handwritten or scanned PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a scanned handwritten response or a scanned document with printed text.",
    )

    conversion_mode = st.radio(
        "Conversion path",
        [
            "Scanned or printed text",
            "Handwritten student response",
        ],
        index=0,
        help="Choose the path that best matches the document you are uploading.",
    )

    if conversion_mode == "Scanned or printed text":
        st.success(
            "This path uses traditional OCR with Local Tesseract. "
            "It is the correct first choice for printed worksheets, typed pages, and most clean scans."
        )
    else:
        st.warning(
            "This path uses AI OCR through OpenAI Vision and is the preferred choice for handwritten student work."
        )

    st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. Actions")

    convert_only = st.button("Convert PDF to text", use_container_width=True, type="primary")
    st.caption(
        "Printed/scanned path uses Local Tesseract. Handwritten path uses OpenAI Vision."
    )

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
        engine_label = "OpenAI Vision" if conversion_mode == "Handwritten student response" else "Local Tesseract"
        log_usage("convert", engine_label)

        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before uploading or processing any PDF.")
        elif handwritten_pdf is None:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Converting PDF to typed text..."):
                pdf_bytes = save_upload_to_bytes(handwritten_pdf)
                result = run_conversion(
                    pdf_bytes=pdf_bytes,
                    original_name=handwritten_pdf.name,
                    conversion_mode=conversion_mode,
                    api_key=api_key,
                    vision_model=vision_model,
                )
                st.session_state.converted_text = result.typed_text
                st.session_state.printable_text = result.printable_text
                st.session_state.conversion_report = result.report_text
            st.success("Conversion complete.")
            st.rerun()

    if compare_notes:
        log_usage("compare", text_model)

        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to compare.")
        elif not notes_text:
            st.error("Please upload or paste study notes text.")
        else:
            with st.spinner("Comparing student response with notes..."):
                result = run_notes_compare(
                    student_text=st.session_state.converted_text,
                    notes_text=notes_text,
                    api_key=api_key,
                )
                st.session_state.compare_report = result.report_text
            st.success("Notes comparison complete.")
            st.rerun()

    if mark_work:
        log_usage("mark", text_model)

        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text to mark.")
        elif not criteria_text:
            st.error("Please upload or paste criteria / rubric text.")
        else:
            with st.spinner("Marking response..."):
                result = run_marking(
                    student_text=st.session_state.converted_text,
                    criteria_text=criteria_text,
                    year_level=year_level,
                    api_key=api_key,
                    notes_text=notes_text if notes_text else None,
                )
                st.session_state.mark_report = result.report_text
            st.success("Marking complete.")
            st.rerun()

    if make_feedback:
        log_usage("feedback", text_model)

        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before processing any PDF content.")
        elif not st.session_state.converted_text:
            st.error("Convert the PDF first so there is student text for feedback.")
        elif not criteria_text:
            st.error("Please upload or paste criteria / rubric text.")
        else:
            with st.spinner("Generating feedback..."):
                result = run_feedback(
                    student_text=st.session_state.converted_text,
                    criteria_text=criteria_text,
                    year_level=year_level,
                    api_key=api_key,
                    text_model=text_model,
                )
                st.session_state.feedback_report = result.report_text
            st.success("Feedback complete.")
            st.rerun()

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
            height=320,
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
            height=320,
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
            height=320,
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
            height=320,
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
            height=320,
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
            height=320,
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
    "Demo shell for Tidy Text Suite. Proprietary OCR, marking, and comparison logic remain server-side. "
    "Results must always be checked by a human reviewer."
)