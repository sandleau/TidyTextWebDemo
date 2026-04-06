import re
import json
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI

try:
    import keyring
except ImportError:
    keyring = None

MODEL = "gpt-5.4-mini"
MIN_RUN = 4
APP_NAME = "Tidy Text — Writing Marker"
KEYRING_SERVICE = "Tidy_Text_Writing_Marker"
KEYRING_USERNAME = "openai_api_key"
APP_DIR = Path.home() / ".tidy_text_writing_marker"
CONFIG_FILE = APP_DIR / "config.json"

client = None


# ----------------------------
# INTRO / ONBOARDING
# ----------------------------

def show_intro():
    print("""
=== Tidy Text — Writing Marker ===

This tool helps teachers:
• Mark student responses using assessment criteria
• Detect possible over-reliance on study notes (optional)

IMPORTANT — Student Privacy
Before using this tool:
• Remove or redact student names and identifying information
• Do not include personal details in uploaded files

OpenAI API Key (Required)
This app uses your own OpenAI API key.

Get your key here:
https://platform.openai.com/api-keys

Steps:
1. Create an OpenAI account (if needed)
2. Generate an API key
3. Paste it into the app when prompted

Your API key:
• Can be stored securely in your system credential manager
• Or stored in a local app folder on your device
• Is NOT shared by this app
• May incur usage costs on your OpenAI account
""")

    if keyring is None:
        print("""
Note:
Secure system storage is currently unavailable because the 'keyring' package is not installed.

To enable secure storage, run this once in Terminal:
pip install keyring

You can still use local app folder storage for now.
""")

    input("Press Enter to continue...")


# ----------------------------
# API KEY STORAGE
# ----------------------------

def ensure_app_dir():
    APP_DIR.mkdir(parents=True, exist_ok=True)

def load_local_config():
    ensure_app_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_local_config(data):
    ensure_app_dir()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_api_key_from_keyring():
    if keyring is None:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None

def save_api_key_to_keyring(api_key):
    if keyring is None:
        return False, "keyring package not installed"
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)
        return True, "Saved to credential manager"
    except Exception as e:
        return False, str(e)

def delete_api_key_from_keyring():
    if keyring is None:
        return
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        pass

def get_api_key_from_local():
    config = load_local_config()
    return config.get("openai_api_key")

def save_api_key_to_local(api_key):
    config = load_local_config()
    config["openai_api_key"] = api_key
    save_local_config(config)

def delete_api_key_from_local():
    config = load_local_config()
    if "openai_api_key" in config:
        del config["openai_api_key"]
        save_local_config(config)

def choose_api_key_storage(api_key):
    print("\nChoose where to store your API key:")
    print("1. System credential manager (recommended)")
    print("2. Local app folder")
    choice = input("Enter 1 or 2 [1]: ").strip() or "1"

    if choice == "1":
        if keyring is None:
            print("""
Secure storage requires the 'keyring' package.

To enable secure storage, run this once in Terminal:

pip install keyring

For now, your API key will be stored in the local app folder instead.
""")
            save_api_key_to_local(api_key)
            return "local"

        ok, msg = save_api_key_to_keyring(api_key)
        if ok:
            print("API key saved securely in your system credential manager.")
            return "keyring"
        else:
            print(f"Could not save to credential manager: {msg}")
            print("Falling back to local app folder storage.")
            save_api_key_to_local(api_key)
            return "local"

    save_api_key_to_local(api_key)
    print("API key saved in local app folder.")
    return "local"

def get_stored_api_key():
    api_key = get_api_key_from_keyring()
    if api_key:
        return api_key, "credential manager"

    api_key = get_api_key_from_local()
    if api_key:
        return api_key, "local app folder"

    return None, None

def setup_api_key():
    api_key, source = get_stored_api_key()
    if api_key:
        print(f"\nOpenAI API key loaded from {source}.")
        return api_key

    print("\nNo stored OpenAI API key was found.")
    api_key = input("Paste your OpenAI API key: ").strip()

    while not api_key.startswith("sk-"):
        print("That does not look like a valid OpenAI API key.")
        api_key = input("Paste your OpenAI API key: ").strip()

    choose_api_key_storage(api_key)
    return api_key

def initialize_client():
    global client
    api_key = setup_api_key()
    client = OpenAI(api_key=api_key)


# ----------------------------
# GENERAL HELPERS
# ----------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def sort_key(k):
    m = re.search(r"(\d+)", k)
    return int(m.group(1)) if m else 999

def flag_rank(flag):
    ranks = {
        "LOW": 1,
        "MEDIUM": 2,
        "MEDIUM-HIGH": 3,
        "HIGH": 4
    }
    return ranks.get(flag, 0)


# ----------------------------
# EXAM QUESTION PARSER
# ----------------------------

def parse_exam_questions(text):
    raw = text.replace("\r", "")

    pattern = r"(?im)^\s*(q\s*\d+|question\s*\d+|\d+\.)\s*"
    matches = list(re.finditer(pattern, raw))

    if not matches:
        return {"FULL_TEXT": raw.strip()}

    sections = {}

    for i, match in enumerate(matches):
        start = match.end()
        label = match.group().strip()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:next_start].strip()

        num = re.search(r"\d+", label)
        key = f"Q{num.group()}" if num else label

        sections[key] = content

    return sections


# ----------------------------
# COPY CHECK
# ----------------------------

COPY_CHECK_PROMPT = """
You are a teacher reviewing whether a student has copied too closely from their study notes.

Full Study Notes:
{notes}

Student Answer:
{answer}

Measured Evidence:
- Direct Phrase Overlap: {phrase_overlap_percent}%
- Closest Sentence Similarity: {sentence_similarity_percent}%
- Suggested Concern Band: {suggested_band}
- Matched Phrases:
{matched_phrases}

Important marking rule:
- Students ARE allowed to use the same ideas and subject-specific words.
- Students should NOT copy exact wording, sentence phrasing, or closely follow the same explanation structure.
- Be alert to:
  1. direct copied phrases
  2. close paraphrasing
  3. same examples used in the same order
  4. same sequence of explanation
  5. same sentence structure with only a few words changed

Be fair, but do not be too soft.

Choose exactly one:
LOW
MEDIUM
MEDIUM-HIGH
HIGH

Return in this exact format:
Similarity: LOW / MEDIUM / MEDIUM-HIGH / HIGH
Reason: 2-4 sentences explaining the judgement
"""

def get_matching_runs(notes_words, answer_words, min_run=4):
    matches = []

    for i in range(len(answer_words)):
        for j in range(len(notes_words)):
            run_length = 0
            while (
                i + run_length < len(answer_words)
                and j + run_length < len(notes_words)
                and answer_words[i + run_length] == notes_words[j + run_length]
            ):
                run_length += 1

            if run_length >= min_run:
                matches.append({
                    "start": i,
                    "end": i + run_length,
                    "length": run_length,
                    "phrase": " ".join(answer_words[i:i + run_length])
                })

    return matches

def merge_runs(matches):
    if not matches:
        return []

    matches = sorted(matches, key=lambda x: (x["start"], -x["length"]))
    merged = [matches[0]]

    for current in matches[1:]:
        last = merged[-1]

        if current["start"] <= last["end"]:
            if current["end"] > last["end"]:
                last["end"] = current["end"]
                last["length"] = last["end"] - last["start"]
                if len(current["phrase"].split()) > len(last["phrase"].split()):
                    last["phrase"] = current["phrase"]
        else:
            merged.append(current)

    return merged

def calculate_phrase_overlap(notes_text, answer_text, min_run=4):
    notes_words = normalize_text(notes_text).split()
    answer_words = normalize_text(answer_text).split()

    if not answer_words:
        return 0.0, []

    matches = get_matching_runs(notes_words, answer_words, min_run)
    merged = merge_runs(matches)

    copied_words = sum(m["length"] for m in merged)
    percent = round((copied_words / len(answer_words)) * 100, 1)

    phrases = []
    seen = set()
    for m in merged:
        if m["phrase"] not in seen:
            phrases.append(m["phrase"])
            seen.add(m["phrase"])

    return percent, phrases

def calculate_sentence_similarity(notes_text, answer_text):
    note_sentences = split_sentences(notes_text)
    answer_sentences = split_sentences(answer_text)

    if not note_sentences or not answer_sentences:
        return 0.0

    scores = []

    for ans in answer_sentences:
        ans_norm = normalize_text(ans)
        best = 0

        for note in note_sentences:
            note_norm = normalize_text(note)
            score = SequenceMatcher(None, ans_norm, note_norm).ratio()
            best = max(best, score)

        scores.append(best)

    return round((sum(scores) / len(scores)) * 100, 1) if scores else 0.0

def suggest_band(phrase_overlap, sentence_similarity):
    if phrase_overlap >= 30 or sentence_similarity >= 82:
        return "HIGH"
    if phrase_overlap >= 15 or sentence_similarity >= 68:
        return "MEDIUM-HIGH"
    if phrase_overlap >= 5 or sentence_similarity >= 45:
        return "MEDIUM"
    return "LOW"

def check_copying(notes, answer):
    phrase_percent, phrases = calculate_phrase_overlap(notes, answer, min_run=MIN_RUN)
    sentence_percent = calculate_sentence_similarity(notes, answer)
    band = suggest_band(phrase_percent, sentence_percent)

    phrase_text = "\n".join(f"- {p}" for p in phrases[:10]) or "None"

    prompt = COPY_CHECK_PROMPT.format(
        notes=notes,
        answer=answer,
        phrase_overlap_percent=phrase_percent,
        sentence_similarity_percent=sentence_percent,
        suggested_band=band,
        matched_phrases=phrase_text
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "phrase_overlap": phrase_percent,
        "sentence_similarity": sentence_percent,
        "suggested_band": band,
        "phrases": phrases,
        "ai": response.choices[0].message.content.strip()
    }


# ----------------------------
# MARKING GUIDE EXTRACTION
# ----------------------------

MARKING_GUIDE_PROMPT = """
You are helping a teacher turn an OCR-extracted assessment criteria page into a usable marking guide.

Assessment Criteria Page:
{criteria_text}

Detected Exam Questions:
{question_keys}

Task:
Create a marking guide for each detected exam question.

Requirements:
- Match the rubric as closely as possible
- Infer which criteria belong to each detected question
- Keep the wording strict, teacher-friendly, and suitable for marking
- Include:
  - question
  - max_mark
  - criteria
- If exact wording is unclear, infer a short sensible question from the criteria page
- Output valid JSON only
- Use this exact structure:

{{
  "Q1": {{
    "question": "...",
    "max_mark": 10,
    "criteria": "..."
  }},
  "Q2": {{
    "question": "...",
    "max_mark": 5,
    "criteria": "..."
  }}
}}

Do not include markdown fences.
"""

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def build_marking_guide(criteria_text, question_keys):
    prompt = MARKING_GUIDE_PROMPT.format(
        criteria_text=criteria_text,
        question_keys=", ".join(question_keys)
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return extract_json(raw)


# ----------------------------
# YEAR LEVEL GUIDANCE
# ----------------------------

def get_year_level_guidance(year_level):
    yl = year_level.strip().lower()

    if yl in ["k", "kindy", "kindergarten"]:
        return (
            "Use very early learning expectations. "
            "Accept very simple words, labels, or short phrases. "
            "Focus on basic understanding rather than detailed explanation. "
            "Do not expect structured sentences or technical language."
        )

    try:
        y = int(yl)
    except ValueError:
        return "Use standard classroom expectations for the stated year level."

    if y <= 2:
        return (
            "Use early primary expectations. "
            "Accept short sentences and simple explanations. "
            "Focus on basic understanding and correct ideas rather than detail or technical terms."
        )
    elif y <= 4:
        return (
            "Use middle primary expectations. "
            "Expect simple explanations with some detail. "
            "Students may use basic subject vocabulary but explanations may still be developing."
        )
    elif y <= 6:
        return (
            "Use upper primary expectations. "
            "Expect clear explanations with some supporting detail. "
            "Encourage use of relevant terms, but allow minor gaps in depth or structure."
        )
    elif y <= 8:
        return (
            "Use early secondary expectations. "
            "Allow simpler wording and developing explanations, but require clear understanding. "
            "Students should begin using subject-specific language."
        )
    elif y <= 10:
        return (
            "Use standard secondary expectations. "
            "Expect clear explanations, relevant detail, and appropriate subject terminology."
        )
    else:
        return (
            "Use senior secondary expectations. "
            "Expect strong analysis, precise terminology, structured responses, and well-developed explanations."
        )


# ----------------------------
# MARKING
# ----------------------------

MARKING_PROMPT = """
You are a teacher marking a student response.

Year level:
{year_level}

Year-level guidance:
{year_guidance}

Question:
{question}

Marking Criteria:
{criteria}

Student Answer:
{answer}

Instructions:
- Mark strictly according to the criteria
- Use the mark ranges and descriptors exactly as written
- Do NOT award marks from a higher band unless the response clearly matches that band
- Be realistic for the stated year level
- Avoid over-marking
- Give brief, useful teacher-style feedback
- The mark must be out of {max_mark}

Return exactly in this format:
Mark: X/{max_mark}
Feedback: ...
"""

def mark_response(question, criteria, answer, max_mark, year_level="8"):
    year_guidance = get_year_level_guidance(year_level)

    prompt = MARKING_PROMPT.format(
        year_level=year_level,
        year_guidance=year_guidance,
        question=question,
        criteria=criteria,
        answer=answer,
        max_mark=max_mark
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ----------------------------
# REPORT SAVING
# ----------------------------

def save_report(report_text, exam_file):
    exam_path = Path(exam_file)
    datestamp = datetime.now().strftime("%Y-%m-%d")
    output_name = f"{exam_path.stem}_TT_Report_{datestamp}.txt"
    output_path = exam_path.parent / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path


# ----------------------------
# MAIN
# ----------------------------

def main():
    show_intro()
    initialize_client()

    exam_file = input("Enter exam text filename [student_exam.txt]: ").strip() or "student_exam.txt"

    use_notes = input("Are there study notes to check? (y/n) [y]: ").strip().lower() or "y"
    notes = None
    if use_notes == "y":
        notes_file = input("Enter study notes text filename [student_studynotes.txt]: ").strip() or "student_studynotes.txt"
        with open(notes_file, "r", encoding="utf-8") as f:
            notes = f.read()

    criteria_file = input("Enter marking criteria text filename [marking_criteria.txt]: ").strip() or "marking_criteria.txt"
    year_level = input("Enter year level (e.g. K, 3, 8, 11) [default 8]: ").strip() or "8"

    with open(exam_file, "r", encoding="utf-8") as f:
        exam = f.read()

    with open(criteria_file, "r", encoding="utf-8") as f:
        criteria_text = f.read()

    questions = parse_exam_questions(exam)
    question_keys = sorted(questions.keys(), key=sort_key)

    print("\nBuilding marking guide from criteria page...\n")
    marking_guide = build_marking_guide(criteria_text, question_keys)

    print("=== INFERRED MARKING GUIDE ===\n")
    inferred_json = json.dumps(marking_guide, indent=2, ensure_ascii=False)
    print(inferred_json)
    print("\n" + "=" * 60 + "\n")

    proceed = input("Proceed with marking using this inferred guide? (y/n) [y]: ").strip().lower() or "y"
    if proceed != "y":
        print("Stopped so you can review or edit the criteria text first.")
        return

    print("\n=== COMBINED ASSESSMENT REPORT ===\n")

    report_lines = []
    report_lines.append("=== TIDY TEXT — WRITING MARKER REPORT ===")
    report_lines.append("")
    report_lines.append(f"Exam file: {exam_file}")
    report_lines.append(f"Criteria file: {criteria_file}")
    report_lines.append(f"Year level: {year_level}")
    report_lines.append(f"Study notes used: {'Yes' if use_notes == 'y' else 'No'}")
    report_lines.append("")
    report_lines.append("=== INFERRED MARKING GUIDE ===")
    report_lines.append("")
    for q, data in marking_guide.items():
        report_lines.append(f"{q} ({data['max_mark']} marks)")
        report_lines.append(data["question"])
        report_lines.append("")
        report_lines.append("Criteria:")
        report_lines.append("")

        # Clean and split criteria nicely
        clean_criteria = data["criteria"].replace("Marks ", "").replace("Mark ", "")
        criteria_parts = clean_criteria.split(". ")

        for part in criteria_parts:
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

    for key in question_keys:
        answer = questions[key]

        if key not in marking_guide:
            section_text = f"{key}\n\nNo inferred marking guide for this question.\n"
            print(section_text)
            print("-" * 60 + "\n")
            report_lines.append(section_text)
            report_lines.append("")
            report_lines.append("-" * 60)
            report_lines.append("")
            continue

        guide = marking_guide[key]

        mark_result = mark_response(
            question=guide["question"],
            criteria=guide["criteria"],
            answer=answer,
            max_mark=guide["max_mark"],
            year_level=year_level
        )

        awarded = 0
        total_possible += int(guide["max_mark"])
        match = re.search(rf"Mark:\s*(\d+)\s*/\s*{guide['max_mark']}", mark_result)
        if match:
            awarded = int(match.group(1))
            total_awarded += awarded

        section_lines = [
            f"{key}",
            "",
            mark_result,
            ""
        ]

        print(f"{key}")
        print(mark_result)

        if use_notes == "y" and notes:
            copy_result = check_copying(notes, answer)

            ai_text = copy_result["ai"]
            sim_match = re.search(r"Similarity:\s*(LOW|MEDIUM|MEDIUM-HIGH|HIGH)", ai_text, re.I)
            flag = sim_match.group(1).upper() if sim_match else copy_result["suggested_band"]

            if flag_rank(flag) > flag_rank(highest_flag):
                highest_flag = flag

            print(f"Copy Check Band: {flag}")
            print(f"Phrase Overlap: {copy_result['phrase_overlap']}%")
            print(f"Sentence Similarity: {copy_result['sentence_similarity']}%")
            print("Copy Check Review:")
            print(ai_text)

            section_lines.append(f"Copy Check Band: {flag}")
            section_lines.append("")
            section_lines.append(f"Phrase Overlap: {copy_result['phrase_overlap']}%")
            section_lines.append(f"Sentence Similarity: {copy_result['sentence_similarity']}%")
            section_lines.append("")
            section_lines.append("Copy Check Review:")
            section_lines.append(ai_text)
            section_lines.append("")

        print("\n" + "-" * 60 + "\n")

        report_lines.extend(section_lines)
        report_lines.append("-" * 60)
        report_lines.append("")

    total_text = f"TOTAL: {total_awarded}/{total_possible}"
    print(total_text)
    report_lines.append(total_text)

    if use_notes == "y":
        copy_flag_text = f"OVERALL COPY FLAG: {highest_flag}"
        print(copy_flag_text)
        report_lines.append("")
        report_lines.append(copy_flag_text)

    report_text = "\n".join(report_lines)
    output_path = save_report(report_text, exam_file)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()