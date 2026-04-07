from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from openai import OpenAI

import tidy_text_v2 as backend
from engines.converter_engine import ConverterJob, run_converter_job
from engines.marker_engine import MarkerJob, run_marker_job


APP_NAME = "Tidy Text Suite"
APP_VERSION = "0.8.0"
APP_TAGLINE = "AI-powered OCR, marking, copy checking, and feedback"

# Change these if your desktop build uses different working model names
DEFAULT_VISION_MODEL = "gpt-5.4-mini"
DEFAULT_TEXT_MODEL = "gpt-5.4-mini"

# Limiter settings
USAGE_LIMIT_FILE = "usage_limits.json"
AI_LIMIT_COUNT = 15
AI_LIMIT_DAYS = 1

# Optional links
FEEDBACK_FORM_URL = "https://forms.gle/your-feedback-form"
PAYHIP_CREDITS_URL = "https://payhip.com/your-credits-product"


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class ConversionResult:
    typed_text: str
    printable_text: str
    report_text: str


@dataclass
class TextResult:
    report_text: str


# -----------------------------
# Session defaults
# -----------------------------
SESSION_DEFAULTS = {
    "converted_text": "",
    "conversion_report": "",
    "compare_report": "",
    "assessment_report": "",
    "feedback_report": "",
    "exam_text_override": "",
    "notes_text_input": "",
    "criteria_text_input": "",
    "session_api_key": "",
    "api_key_source_label": "not set",
    "current_base_name": "TTS_Output",
    "client_session_id": str(uuid.uuid4()),
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Helpers
# -----------------------------
def sanitize_stem(name: str) -> str:
    stem = Path(name).stem if name else "TTS_Output"
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or "TTS_Output"


def timestamped_filename(base_name: str, suffix: str, ext: str = ".txt") -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_base = sanitize_stem(base_name)
    return f"{safe_base}_{suffix}_{stamp}{ext}"


def save_upload_to_bytes(uploaded_file) -> bytes:
    return uploaded_file.getvalue() if uploaded_file is not None else b""


def save_text_to_tempfile(text: str, suffix: str = ".txt") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tmp:
        tmp.write(text)
        return tmp.name


def download_text_button(label: str, text: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )


def get_api_key() -> tuple[Optional[str], str]:
    """
    Priority:
    1. Session override entered by user
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
    Basic usage logger without storing document content.
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


def extract_copy_band(ai_text: str, fallback: str = "LOW") -> str:
    match = re.search(r"Similarity:\s*(LOW|MEDIUM|MEDIUM-HIGH|HIGH)", ai_text, re.I)
    return match.group(1).upper() if match else fallback


def resolve_exam_text(exam_text_file, exam_text_manual: str) -> str:
    """
    Important behavior:
    This does not fall back to OCR output.
    Users must explicitly paste/upload the exam text they want processed.
    """
    if exam_text_file is not None:
        return exam_text_file.getvalue().decode("utf-8", errors="replace")
    if exam_text_manual.strip():
        return exam_text_manual.strip()
    return ""


def resolve_notes_text(notes_file, notes_text_manual: str) -> str:
    if notes_file is not None:
        return notes_file.getvalue().decode("utf-8", errors="replace")
    if notes_text_manual.strip():
        return notes_text_manual.strip()
    return ""


def resolve_criteria_text(criteria_file, criteria_text_manual: str) -> str:
    if criteria_file is not None:
        return criteria_file.getvalue().decode("utf-8", errors="replace")
    if criteria_text_manual.strip():
        return criteria_text_manual.strip()
    return ""


def show_optional_link_button(label: str, url: str, help_text: str = "") -> None:
    if url and "your-" not in url and "forms.gle/your" not in url:
        st.link_button(label, url, use_container_width=True)
    else:
        st.caption(help_text)


# -----------------------------
# Hybrid limiter helpers
# -----------------------------
def get_client_ip() -> str:
    try:
        return requests.get("https://api.ipify.org", timeout=5).text
    except Exception:
        return "unknown"


def load_usage_limit_data() -> list[dict]:
    path = Path(USAGE_LIMIT_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_usage_limit_data(data: list[dict]) -> None:
    Path(USAGE_LIMIT_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prune_old_usage_entries(data: list[dict]) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=AI_LIMIT_DAYS)
    kept = []
    for row in data:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts >= cutoff:
                kept.append(row)
        except Exception:
            continue
    return kept


def count_recent_ai_usage(ip: str, session_id: str) -> tuple[int, int]:
    data = prune_old_usage_entries(load_usage_limit_data())
    ip_count = 0
    session_count = 0

    for row in data:
        if row.get("kind") != "ai_action":
            continue
        if row.get("ip") == ip:
            ip_count += 1
        if row.get("session_id") == session_id:
            session_count += 1

    return ip_count, session_count


def can_use_shared_ai_key() -> tuple[bool, str]:
    """
    Shared/server key is limited.
    User-supplied session keys are not limited.
    """
    key_source = st.session_state.get("api_key_source_label", "not set")

    if key_source == "session key":
        return True, "Using your own session API key. Shared demo limiter is bypassed."

    ip = get_client_ip()
    session_id = st.session_state.get("client_session_id", "unknown")

    ip_count, session_count = count_recent_ai_usage(ip, session_id)

    if ip_count >= AI_LIMIT_COUNT or session_count >= AI_LIMIT_COUNT:
        return (
            False,
            "Free shared demo limit reached for AI actions in the current 24-hour window. Please try again later, use your own API key in the sidebar, or buy credits when available.",
        )

    remaining = AI_LIMIT_COUNT - max(ip_count, session_count)
    return True, f"Shared AI actions remaining in the current 24-hour window: {remaining}"


def record_ai_usage(action: str) -> None:
    key_source = st.session_state.get("api_key_source_label", "not set")

    # Do not track/limit user-supplied keys for quota purposes
    if key_source == "session key":
        return

    ip = get_client_ip()
    session_id = st.session_state.get("client_session_id", "unknown")

    data = prune_old_usage_entries(load_usage_limit_data())
    data.append(
        {
            "timestamp": datetime.now().isoformat(),
            "kind": "ai_action",
            "action": action,
            "ip": ip,
            "session_id": session_id,
        }
    )
    save_usage_limit_data(data)


# -----------------------------
# Core functions
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
        pdf_path = tmp.name

    if conversion_mode == "Handwritten student response":
        engine_name = "OpenAI Vision"
        model_name = vision_model
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key is available for handwriting OCR. Add one in the sidebar for this session, or configure OPENAI_API_KEY."
            )
    else:
        engine_name = "Local Tesseract"
        model_name = ""
        api_key = None

    job = ConverterJob(
        pdf_path=pdf_path,
        output_folder=tempfile.gettempdir(),
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


def run_notes_compare(student_text: str, notes_text: str, api_key: Optional[str]) -> TextResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for notes comparison. Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    backend.client = OpenAI(api_key=api_key)
    copy_result = backend.check_copying(notes_text, student_text)

    ai_text = copy_result["ai"]
    band = extract_copy_band(ai_text, copy_result.get("suggested_band", "LOW"))

    lines = [
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

    return TextResult(report_text="\n".join(lines))


def build_overall_teacher_comment(
    student_text: str,
    criteria_text: str,
    assessment_report: str,
    year_level: str,
    api_key: Optional[str],
    text_model: str,
) -> str:
    if not api_key:
        return "No API key available for overall teacher comment."

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are helping a teacher write a concise overall teacher comment for an assessment report.

Year level: {year_level}

Write a short teacher-facing summary in plain text with these headings:
Overall strengths
Main concerns
Next steps

Requirements:
- Keep it concise.
- Use professional teacher language.
- Do not repeat the full report.
- Do not invent marks.
- Base the comment on the supplied report, student response, and criteria.

STUDENT RESPONSE:
{student_text}

CRITERIA:
{criteria_text}

ASSESSMENT REPORT:
{assessment_report}
""".strip()

    response = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": "You write concise, teacher-facing assessment summaries in plain text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def append_percentage_and_comment(
    report_text: str,
    student_text: str,
    criteria_text: str,
    year_level: str,
    api_key: Optional[str],
    text_model: str,
) -> str:
    awarded = None
    possible = None

    match = re.search(r"TOTAL:\s*(\d+)\s*/\s*(\d+)", report_text)
    if match:
        awarded = int(match.group(1))
        possible = int(match.group(2))

    extra_lines = ["", "=" * 60, ""]

    if awarded is not None and possible:
        percentage = round((awarded / possible) * 100, 1)
        extra_lines.extend(
            [
                "=== OVERALL RESULT SUMMARY ===",
                "",
                f"Total Score: {awarded}/{possible}",
                f"Percentage: {percentage}%",
                "",
            ]
        )

    teacher_comment = build_overall_teacher_comment(
        student_text=student_text,
        criteria_text=criteria_text,
        assessment_report=report_text,
        year_level=year_level,
        api_key=api_key,
        text_model=text_model,
    )

    extra_lines.extend(
        [
            "=== OVERALL TEACHER COMMENT ===",
            "",
            teacher_comment,
        ]
    )

    return report_text + "\n" + "\n".join(extra_lines)


def run_assessment_report(
    exam_text: str,
    criteria_text: str,
    notes_text: str,
    year_level: str,
    api_key: Optional[str],
    text_model: str,
) -> TextResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for marking. Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    exam_path = save_text_to_tempfile(exam_text)
    criteria_path = save_text_to_tempfile(criteria_text)
    notes_path = save_text_to_tempfile(notes_text) if notes_text.strip() else None

    job = MarkerJob(
        exam_file=exam_path,
        criteria_file=criteria_path,
        notes_file=notes_path,
        year_level=year_level,
        use_notes=bool(notes_text.strip()),
        api_key=api_key,
    )

    result = run_marker_job(job)
    if not result.success:
        raise RuntimeError(result.error)

    enhanced_report = append_percentage_and_comment(
        report_text=result.report_text,
        student_text=exam_text,
        criteria_text=criteria_text,
        year_level=year_level,
        api_key=api_key,
        text_model=text_model,
    )

    return TextResult(report_text=enhanced_report)


def run_feedback(
    student_text: str,
    criteria_text: str,
    year_level: str,
    api_key: Optional[str],
    text_model: str,
) -> TextResult:
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key is available for feedback generation. Add one in the sidebar for this session, or configure OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are helping a teacher write feedback on student work.

Year level: {year_level}

Write plain-text feedback with these headings:
Strengths
Next steps
Suggested improvement actions

Requirements:
- Keep it concise and practical.
- Use teacher-friendly language.
- Do not invent marks.
- Base the feedback only on the student response and criteria.

STUDENT RESPONSE:
{student_text}

CRITERIA:
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

    return TextResult(report_text="\n".join(report_lines))


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="📝",
    layout="wide",
)

api_key, api_key_source = get_api_key()
st.session_state["api_key_source_label"] = api_key_source


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Workflow")
    st.markdown(
        "**Printed / scanned text**\n"
        "Upload PDF → Convert with traditional OCR → Review in output panel → Copy/paste the text you want into Exam text → Process\n\n"
        "**Handwritten student work**\n"
        "Upload PDF → Convert with AI OCR → Review in output panel → Copy/paste the text you want into Exam text → Process"
    )

    st.subheader("OpenAI settings")
    st.text_input(
        "Session API key override",
        type="password",
        key="session_api_key",
        help="Optional. Paste your own OpenAI API key for this browser session if the default key is unavailable or out of credit.",
    )

    api_key, api_key_source = get_api_key()
    st.session_state["api_key_source_label"] = api_key_source

    if api_key:
        st.success(f"API key available via {api_key_source}.")
    else:
        st.warning("No OpenAI API key detected yet.")

    allowed, limit_message = can_use_shared_ai_key()
    if st.session_state.get("api_key_source_label") == "session key":
        st.info(limit_message)
    else:
        st.caption(limit_message)

    with st.expander("How to use your own API key for this session", expanded=False):
        st.markdown(
            """
1. Paste your OpenAI API key into **Session API key override** above.  
2. It will be used only for this current Streamlit session.  
3. Handwritten OCR, notes comparison, marking, and feedback will then use that key.  
4. Printed/scanned OCR does not need an API key because it uses Local Tesseract.
"""
        )

    st.subheader("Buy credits")
    st.caption(
        "Need more AI runs today? A paid credits system is planned for a later release. For now, you can use your own API key for unlimited personal use in this session."
    )
    show_optional_link_button(
        "Buy credits (coming soon)",
        PAYHIP_CREDITS_URL,
        help_text="Add your Payhip credits product link here later to enable a clear upgrade path.",
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
        "Keep long-term API keys in Streamlit secrets or environment variables. Do not hardcode them into this file."
    )


# -----------------------------
# Header
# -----------------------------
st.title(APP_NAME)
st.caption(f"v{APP_VERSION} • {APP_TAGLINE}")

st.warning(
    "Privacy warning: Do not upload PDFs that contain private or identifying student information. Best practice is to remove, redact, or exclude names, student numbers, addresses, date of birth, school IDs, or any other identifying details before upload."
)

st.info(
    "Recommended workflow: Convert the PDF first, review the OCR text in the output panel, then paste the final text you want assessed into the Exam text field."
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
    "I confirm that any uploaded PDF has been checked and does not contain private or identifying student information, and does not include unlawful, offensive, or inappropriate material (including hate speech, abuse, or illegal content).",
    value=False,
)

st.caption(
    "© Sandle Software — Tidy Text Suite. All rights reserved. This software, including all underlying logic, workflows, and outputs, is the intellectual property of Sandle Software and may not be copied, reproduced, reverse engineered, or redistributed without permission."
)


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # -------------------------
    # Step 1: Upload + Convert
    # -------------------------
    st.subheader("Step 1: Upload and convert")
    pdf_file = st.file_uploader(
        "Student handwritten or scanned PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a scanned handwritten response or a scanned document with printed text.",
    )

    conversion_mode = st.radio(
        "Conversion path",
        ["Scanned or printed text", "Handwritten student response"],
        index=0,
        help="Choose the path that best matches the document you are uploading.",
    )

    if conversion_mode == "Scanned or printed text":
        st.success(
            "This path uses Local Tesseract and is best for printed worksheets, typed pages, and most clean scans."
        )
    else:
        st.warning(
            "This path uses OpenAI Vision and is the preferred choice for handwritten student work."
        )

    if st.button("Convert PDF to text", use_container_width=True, type="primary"):
        engine_label = "OpenAI Vision" if conversion_mode == "Handwritten student response" else "Local Tesseract"
        log_usage("convert", engine_label)

        if not privacy_confirmed:
            st.error("Please confirm the privacy checkbox before uploading or processing any PDF.")
        elif pdf_file is None:
            st.error("Please upload a PDF first.")
        elif conversion_mode == "Handwritten student response":
            allowed, limit_message = can_use_shared_ai_key()
            if not allowed:
                st.error(limit_message)
                st.stop()
        else:
            pass

        with st.spinner("Converting PDF to typed text..."):
            pdf_bytes = save_upload_to_bytes(pdf_file)
            result = run_conversion(
                pdf_bytes=pdf_bytes,
                original_name=pdf_file.name,
                conversion_mode=conversion_mode,
                api_key=api_key,
                vision_model=vision_model,
            )

            st.session_state["converted_text"] = result.typed_text
            st.session_state["conversion_report"] = result.report_text
            st.session_state["current_base_name"] = sanitize_stem(pdf_file.name)

            # Clear downstream outputs after reconversion
            st.session_state["compare_report"] = ""
            st.session_state["assessment_report"] = ""
            st.session_state["feedback_report"] = ""

            if conversion_mode == "Handwritten student response":
                record_ai_usage("handwriting_ocr")

        st.success(
            "Conversion complete. Review the OCR in the output panel, then paste the final text you want assessed into the Exam text field below."
        )
        st.rerun()

    # -------------------------
    # Step 2: Add / override text inputs
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Step 2: Add text for processing")
    st.caption(
        "OCR output does not auto-fill the Exam text field. Paste or upload only the exact exam text you want used for comparison, marking, and feedback."
    )

    exam_text_file = st.file_uploader(
        "Exam text (TXT/MD)",
        type=["txt", "md"],
        accept_multiple_files=False,
        key="exam_text_file",
    )

    st.text_area(
        "Exam text",
        key="exam_text_override",
        height=180,
        placeholder="Paste the final exam response text here for processing.",
    )

    if st.session_state["converted_text"].strip() and not st.session_state["exam_text_override"].strip() and exam_text_file is None:
        st.info(
            "Converted OCR text is available in the Outputs → Converted text tab. Copy the text you want from there, then paste it into the Exam text field here."
        )

    notes_file = st.file_uploader(
        "Study notes text (TXT/MD)",
        type=["txt", "md"],
        accept_multiple_files=False,
        key="notes_text_file",
    )

    st.text_area(
        "Study notes text",
        key="notes_text_input",
        height=140,
        placeholder="Paste study notes here if not uploading a text file.",
    )

    criteria_file = st.file_uploader(
        "Criteria / rubric text (TXT/MD)",
        type=["txt", "md"],
        accept_multiple_files=False,
        key="criteria_text_file",
    )

    st.text_area(
        "Criteria / rubric text",
        key="criteria_text_input",
        height=140,
        placeholder="Paste criteria or rubric here if not uploading a text file.",
    )

    exam_text = resolve_exam_text(
        exam_text_file=exam_text_file,
        exam_text_manual=st.session_state["exam_text_override"],
    )
    notes_text = resolve_notes_text(
        notes_file=notes_file,
        notes_text_manual=st.session_state["notes_text_input"],
    )
    criteria_text = resolve_criteria_text(
        criteria_file=criteria_file,
        criteria_text_manual=st.session_state["criteria_text_input"],
    )

    # -------------------------
    # Step 3: Process
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Step 3: Process")
    st.caption("Use the Exam text field above for notes comparison, assessment reporting, and feedback.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Compare with notes", use_container_width=True):
            log_usage("compare", text_model)

            if not privacy_confirmed:
                st.error("Please confirm the privacy checkbox before processing any PDF content.")
            elif not exam_text.strip():
                st.error("Provide exam text first by pasting or uploading the exam text in Step 2.")
            elif not notes_text.strip():
                st.error("Please upload or paste study notes text.")
            else:
                allowed, limit_message = can_use_shared_ai_key()
                if not allowed:
                    st.error(limit_message)
                    st.stop()

                with st.spinner("Comparing student response with notes..."):
                    result = run_notes_compare(
                        student_text=exam_text,
                        notes_text=notes_text,
                        api_key=api_key,
                    )
                    st.session_state["compare_report"] = result.report_text
                    record_ai_usage("compare")

                st.success("Notes comparison complete.")
                st.rerun()

    with col_b:
        if st.button("Generate assessment report", use_container_width=True):
            log_usage("assessment_report", text_model)

            if not privacy_confirmed:
                st.error("Please confirm the privacy checkbox before processing any PDF content.")
            elif not exam_text.strip():
                st.error("Provide exam text first by pasting or uploading the exam text in Step 2.")
            elif not criteria_text.strip():
                st.error("Please upload or paste criteria / rubric text.")
            else:
                allowed, limit_message = can_use_shared_ai_key()
                if not allowed:
                    st.error(limit_message)
                    st.stop()

                with st.spinner("Generating assessment report..."):
                    result = run_assessment_report(
                        exam_text=exam_text,
                        criteria_text=criteria_text,
                        notes_text=notes_text,
                        year_level=year_level,
                        api_key=api_key,
                        text_model=text_model,
                    )
                    st.session_state["assessment_report"] = result.report_text
                    record_ai_usage("assessment_report")

                st.success("Assessment report complete.")
                st.rerun()

    with col_c:
        if st.button("Generate feedback", use_container_width=True):
            log_usage("feedback", text_model)

            if not privacy_confirmed:
                st.error("Please confirm the privacy checkbox before processing any PDF content.")
            elif not exam_text.strip():
                st.error("Provide exam text first by pasting or uploading the exam text in Step 2.")
            elif not criteria_text.strip():
                st.error("Please upload or paste criteria / rubric text.")
            else:
                allowed, limit_message = can_use_shared_ai_key()
                if not allowed:
                    st.error(limit_message)
                    st.stop()

                with st.spinner("Generating feedback..."):
                    result = run_feedback(
                        student_text=exam_text,
                        criteria_text=criteria_text,
                        year_level=year_level,
                        api_key=api_key,
                        text_model=text_model,
                    )
                    st.session_state["feedback_report"] = result.report_text
                    record_ai_usage("feedback")

                st.success("Feedback complete.")
                st.rerun()


with right_col:
    st.subheader("Outputs")

    if not any(
        [
            st.session_state["converted_text"].strip(),
            st.session_state["compare_report"].strip(),
            st.session_state["assessment_report"].strip(),
            st.session_state["feedback_report"].strip(),
        ]
    ):

            st.info("""
            **Start here (Quick test):**
            1. Upload ONE student PDF
            2. Click Convert
            3. Copy into Exam Text
            4. Click Generate Report

👉              Just try one student work sample first.
            """)
            
            st.info(
                 "Full workflow:\n"
                 "1. Upload and convert a PDF.\n"
                 "2. Review the OCR in the output panel.\n"
                 "3. Paste the final exam text into the Exam text field.\n"
                 "4. Add study notes and/or criteria.\n"
                 "5. Generate output and download it below.\n"
            )

    with st.expander("Suggested scanning practice", expanded=False):
        st.markdown(
            """
- Scan the written exam response and study notes as **two separate PDF files** before converting to text.
- Remove, cover, or crop out **student names, school names, and other identifying details** before upload.
- Use **simple anonymous file names** and keep your own separate note if you need to match files back to students safely.
- After converting a **criteria or rubric** document to text, quickly review and tidy the formatting if needed so the marking criteria and performance levels are easy for the AI to interpret.
"""
        )

    st.warning(
        "Download outputs as you go. This demo shows current results in-session only, and later runs may replace what is currently displayed in the output tabs."
    )

    output_tabs = st.tabs(
        [
            "Converted text",
            "Processing report",
            "Notes compare",
            "Assessment report",
            "Feedback report",
        ]
    )

    base_name = st.session_state["current_base_name"]

    with output_tabs[0]:
        st.text_area(
            "Converted text output",
            value=st.session_state["converted_text"],
            height=320,
        )
        if st.session_state["converted_text"].strip():
            download_text_button(
                "Download converted text",
                st.session_state["converted_text"],
                timestamped_filename(base_name, "TTS_Converted"),
            )

    with output_tabs[1]:
        st.text_area(
            "Processing / conversion report",
            value=st.session_state["conversion_report"],
            height=320,
        )
        if st.session_state["conversion_report"].strip():
            download_text_button(
                "Download conversion report",
                st.session_state["conversion_report"],
                timestamped_filename(base_name, "TTS_Conversion_Report"),
            )

    with output_tabs[2]:
        st.text_area(
            "Notes comparison report",
            value=st.session_state["compare_report"],
            height=320,
        )
        if st.session_state["compare_report"].strip():
            download_text_button(
                "Download notes comparison report",
                st.session_state["compare_report"],
                timestamped_filename(base_name, "TTS_Compare"),
            )

    with output_tabs[3]:
        st.text_area(
            "Assessment report",
            value=st.session_state["assessment_report"],
            height=360,
        )
        if st.session_state["assessment_report"].strip():
            download_text_button(
                "Download assessment report",
                st.session_state["assessment_report"],
                timestamped_filename(base_name, "TTS_Assessment_Report"),
            )

    with output_tabs[4]:
        st.text_area(
            "Feedback report",
            value=st.session_state["feedback_report"],
            height=320,
        )
        if st.session_state["feedback_report"].strip():
            download_text_button(
                "Download feedback report",
                st.session_state["feedback_report"],
                timestamped_filename(base_name, "TTS_Feedback"),
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Optional feedback")
    st.caption("If this demo helps or something goes wrong, feedback is welcome.")
    show_optional_link_button(
        "Open feedback form",
        FEEDBACK_FORM_URL,
        help_text="https://forms.gle/2ktPa1hA9e8Xe2feA",
    )

st.divider()
st.caption(
    "Demo shell for Tidy Text Suite. Proprietary OCR, marking, and comparison logic remain server-side. Results must always be checked by a human reviewer."
)
