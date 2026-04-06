from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import json
import re
import threading

from openai import OpenAI
import tidy_text_v2 as backend


ProgressCallback = Optional[Callable[[str], None]]


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
class MarkerJob:
    exam_file: str
    criteria_file: str
    notes_file: Optional[str]
    year_level: str
    use_notes: bool
    api_key: str


@dataclass
class MarkerResult:
    success: bool
    output_path: Optional[str] = None
    report_text: str = ""
    error: Optional[str] = None


def run_marker_job(
    job: MarkerJob,
    progress_callback: ProgressCallback = None,
    cancel_event: Optional[threading.Event] = None,
) -> MarkerResult:

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Run cancelled by user.")

    try:
        log("Loading API client...\n")
        backend.client = OpenAI(api_key=job.api_key)

        check_cancel()

        log(f"Exam file: {job.exam_file}\n")
        log(f"Criteria file: {job.criteria_file}\n")
        log(f"Year level: {job.year_level}\n")
        log(f"Copy detection: {'Enabled' if job.use_notes else 'Disabled'}\n")
        if job.use_notes and job.notes_file:
            log(f"Study notes file: {job.notes_file}\n")
        log("\n")

        exam = Path(job.exam_file).read_text(encoding="utf-8")
        criteria_text = Path(job.criteria_file).read_text(encoding="utf-8")
        notes = Path(job.notes_file).read_text(encoding="utf-8") if job.use_notes and job.notes_file else None

        check_cancel()

        log("Parsing questions from exam...\n")
        questions = backend.parse_exam_questions(exam)
        question_keys = sorted(questions.keys(), key=backend.sort_key)
        log(f"Detected questions: {', '.join(question_keys)}\n\n")

        check_cancel()

        log("Building inferred marking guide...\n\n")
        marking_guide = backend.build_marking_guide(criteria_text, question_keys)

        log("=== INFERRED MARKING GUIDE ===\n\n")
        log(json.dumps(marking_guide, indent=2, ensure_ascii=False) + "\n\n")
        log("=" * 60 + "\n\n")

        report_lines = []
        report_lines.append("=== TIDY TEXT - WRITING MARKER REPORT ===")
        report_lines.append("")
        report_lines.append(f"Exam file: {job.exam_file}")
        report_lines.append(f"Criteria file: {job.criteria_file}")
        report_lines.append(f"Year level: {job.year_level}")
        report_lines.append(f"Study notes used: {'Yes' if job.use_notes else 'No'}")
        report_lines.append("")
        report_lines.append("=== INFERRED MARKING GUIDE ===")
        report_lines.append("")

        for q, data in marking_guide.items():
            report_lines.append(f"{q} ({data['max_mark']} marks)")
            report_lines.append(data["question"])
            report_lines.append("")
            report_lines.append("Criteria:")
            report_lines.append("")
            clean_criteria = data["criteria"].replace("Marks ", "").replace("Mark ", "")
            for part in clean_criteria.split(". "):
                part = part.strip()
                if part:
                    report_lines.append(part)
            report_lines.append("")
            report_lines.append("-" * 60)
            report_lines.append("")

        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("")
        report_lines.append("=== COMBINED ASSESSMENT REPORT ===")
        report_lines.append("")

        total_awarded = 0
        total_possible = 0
        highest_flag = "LOW"

        log("Starting question-by-question marking...\n\n")

        for idx, key in enumerate(question_keys, start=1):
            check_cancel()

            answer = questions[key]
            log(f"[{idx}/{len(question_keys)}] Marking {key}...\n")

            if key not in marking_guide:
                text = f"{key}\n\nNo inferred marking guide for this question.\n"
                log(f"\n--- {key} — marked ---\n\n{text}\n")
                log("-" * 60 + "\n\n")
                report_lines.append(text)
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("")
                continue

            guide = marking_guide[key]
            mark_result = backend.mark_response(
                question=guide["question"],
                criteria=guide["criteria"],
                answer=answer,
                max_mark=guide["max_mark"],
                year_level=job.year_level
            )

            awarded = 0
            total_possible += int(guide["max_mark"])
            match = re.search(rf"Mark:\s*(\d+)\s*/\s*{guide['max_mark']}", mark_result)
            if match:
                awarded = int(match.group(1))
                total_awarded += awarded

            log(f"\n--- {key} — marked ---\n\n{mark_result}\n\n")

            section_lines = [
                f"{key}",
                "",
                mark_result,
                ""
            ]

            check_cancel()

            if job.use_notes and notes:
                log("Running copy detection...\n")
                copy_result = backend.check_copying(notes, answer)
                ai_text = copy_result["ai"]
                sim_match = re.search(r"Similarity:\s*(LOW|MEDIUM|MEDIUM-HIGH|HIGH)", ai_text, re.I)
                flag = sim_match.group(1).upper() if sim_match else copy_result["suggested_band"]

                if backend.flag_rank(flag) > backend.flag_rank(highest_flag):
                    highest_flag = flag

                log(f"Copy Check Band: {flag}\n")
                log(f"Phrase Overlap: {copy_result['phrase_overlap']}%\n")
                log(f"Sentence Similarity: {copy_result['sentence_similarity']}%\n")
                log("Copy Check Review:\n")
                log(ai_text + "\n")

                section_lines.append(f"Copy Check Band: {flag}")
                section_lines.append("")
                section_lines.append(f"Phrase Overlap: {copy_result['phrase_overlap']}%")
                section_lines.append(f"Sentence Similarity: {copy_result['sentence_similarity']}%")
                section_lines.append("")
                section_lines.append("Copy Check Review:")
                section_lines.append(ai_text)
                section_lines.append("")

            log("-" * 60 + "\n\n")
            report_lines.extend(section_lines)
            report_lines.append("-" * 60)
            report_lines.append("")

        total_text = f"TOTAL: {total_awarded}/{total_possible}"
        log(total_text + "\n")
        report_lines.append(total_text)

        if job.use_notes:
            copy_flag_text = f"OVERALL COPY FLAG: {highest_flag}"
            log(copy_flag_text + "\n")
            report_lines.append("")
            report_lines.append(copy_flag_text)

        report_text = "\n".join(report_lines)
        exam_path = Path(job.exam_file)
        datestamp = datetime.now().strftime("%Y-%m-%d")
        report_name = f"{exam_path.stem}_TT_Report_{datestamp}.txt"
        output_path = get_unique_output_path(exam_path.parent / report_name)
        output_path.write_text(report_text, encoding="utf-8")

        log(f"\nReport saved to: {output_path}\n")

        return MarkerResult(
            success=True,
            output_path=str(output_path),
            report_text=report_text,
        )

    except Exception as e:
        return MarkerResult(success=False, error=str(e))