from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
from openai import OpenAI

import tidy_text_v2 as backend
from engines.converter_engine import ConverterJob, run_converter_job


APP_NAME = "Tidy Text Suite"
APP_VERSION = "0.5.0"
APP_TAGLINE = "AI-powered OCR, marking, and feedback"


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class ConversionResult:
    typed_text: str
    printable_text: str
    report_text: str


@dataclass
class Result:
    report_text: str


# -----------------------------
# API Key
# -----------------------------
def get_api_key():
    if st.session_state.get("session_api_key"):
        return st.session_state["session_api_key"], "session"
    try:
        return st.secrets["OPENAI_API_KEY"], "secrets"
    except Exception:
        return None, "missing"


# -----------------------------
# Conversion
# -----------------------------
def run_conversion(pdf_bytes, filename, mode, api_key):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name

    if mode == "Handwritten":
        engine = "OpenAI Vision"
        if not api_key:
            raise RuntimeError("API key required for handwriting OCR")
    else:
        engine = "Local Tesseract"
        api_key = None

    job = ConverterJob(
        pdf_path=path,
        output_folder=tempfile.gettempdir(),
        output_name="output",
        doc_type="Exam",
        engine=engine,
        model_name="gpt-5.4-mini",
        out_ext=".txt",
        api_key=api_key,
    )

    result = run_converter_job(job)

    return ConversionResult(
        typed_text=result.full_text,
        printable_text=result.full_text,
        report_text=result.full_text,
    )


# -----------------------------
# AI Functions
# -----------------------------
def run_compare(text, notes, api_key):
    backend.client = OpenAI(api_key=api_key)
    return Result(report_text=backend.check_copying(notes, text)["ai"])


def run_mark(text, criteria, year, api_key):
    backend.client = OpenAI(api_key=api_key)
    return Result(
        report_text=backend.mark_response(
            question="Full Response",
            criteria=criteria,
            answer=text,
            max_mark=10,
            year_level=year,
        )
    )


def run_feedback(text, criteria, year, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "Write clear teacher feedback."},
            {"role": "user", "content": f"{text}\n\nCriteria:\n{criteria}"},
        ],
    )

    return Result(report_text=response.choices[0].message.content)


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(layout="wide")

# Session state
for key in ["converted", "compare", "mark", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = ""


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("API Settings")

    st.text_input(
        "Session API key (optional)",
        type="password",
        key="session_api_key",
    )

    api_key, source = get_api_key()

    if api_key:
        st.success(f"Using API key from {source}")
    else:
        st.warning("No API key found")

    year = st.selectbox(
        "Year level",
        ["Default", "7", "8", "9", "10", "11", "12"],
    )


# -----------------------------
# Header
# -----------------------------
st.title(APP_NAME)
st.caption(APP_TAGLINE)

st.warning("Do NOT upload student-identifiable information.")

privacy = st.checkbox("I confirm data is safe to use.")


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])


# =============================
# LEFT SIDE (INPUT)
# =============================
with left:

    # Step 1
    st.header("Step 1: Convert PDF")

    file = st.file_uploader("Upload PDF", type=["pdf"])

    mode = st.radio(
        "Select type",
        ["Scanned", "Handwritten"],
    )

    if st.button("Convert"):
        if not privacy:
            st.error("Confirm privacy first")
        elif not file:
            st.error("Upload a PDF")
        else:
            res = run_conversion(file.read(), file.name, mode, api_key)
            st.session_state.converted = res.typed_text
            st.success("Converted")
            st.rerun()

    # Step 2
    st.header("Step 2: Provide Text")

    exam_text = st.text_area(
        "Exam Text (overrides OCR)",
        value=st.session_state.converted,
        height=200,
    )

    notes = st.text_area("Study Notes", height=150)

    criteria = st.text_area("Criteria / Rubric", height=150)

    # Step 3
    st.header("Step 3: Process")

    if st.button("Compare"):
        st.session_state.compare = run_compare(exam_text, notes, api_key).report_text
        st.rerun()

    if st.button("Mark"):
        st.session_state.mark = run_mark(exam_text, criteria, year, api_key).report_text
        st.rerun()

    if st.button("Feedback"):
        st.session_state.feedback = run_feedback(exam_text, criteria, year, api_key).report_text
        st.rerun()


# =============================
# RIGHT SIDE (OUTPUT)
# =============================
with right:

    st.header("Outputs")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Converted", "Compare", "Mark", "Feedback"]
    )

    with tab1:
        st.text_area("Converted Text", st.session_state.converted, height=300)

    with tab2:
        st.text_area("Compare", st.session_state.compare, height=300)

    with tab3:
        st.text_area("Marking", st.session_state.mark, height=300)

    with tab4:
        st.text_area("Feedback", st.session_state.feedback, height=300)