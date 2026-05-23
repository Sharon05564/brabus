"""
extract_features.py
-------------------
Reusable numeric feature extraction for document text.

This module is intentionally kept simple so that:
  1. Each feature can be explained in one sentence during an interview.
  2. The exact same feature logic can be mirrored in Swift (DocumentFeatureExtractor.swift).
  3. The feature vector can be fed directly into a PyTorch / Core ML numeric model.

Feature order (must match DocumentFeatureExtractor.swift and brabus_pytorch_metadata.json):
  0  character_count
  1  word_count
  2  sentence_count
  3  date_count
  4  email_count
  5  phone_count
  6  dollar_amount_count
  7  action_word_count
  8  education_keyword_count
  9  job_keyword_count
 10  receipt_keyword_count
 11  deadline_keyword_count
"""

import re
from typing import Union

# ---------------------------------------------------------------------------
# Keyword lists used by multiple scripts — centralised here so Swift and
# Python always reference the same vocabulary.
# ---------------------------------------------------------------------------

ACTION_WORDS = [
    "submit", "apply", "attend", "register", "pay", "email",
    "upload", "bring", "complete", "review", "contact", "schedule",
    "confirm", "rsvp", "download", "print", "sign", "renew",
    "prepare", "deliver", "return", "visit", "follow", "update",
]

EDUCATION_KEYWORDS = [
    "syllabus", "lecture", "exam", "midterm", "final", "assignment",
    "homework", "quiz", "grade", "professor", "instructor", "course",
    "credit", "semester", "academic", "university", "college", "student",
    "rubric", "transcript", "gpa", "attendance", "textbook", "tuition",
    "enrollment", "prerequisite", "capstone", "thesis", "dissertation",
]

JOB_KEYWORDS = [
    "resume", "internship", "position", "qualifications", "responsibilities",
    "salary", "compensation", "hiring", "candidate", "interview",
    "application", "employer", "full-time", "part-time", "career",
    "skills", "experience", "role", "benefits", "recruiter",
    "cover letter", "linkedin", "job", "opening", "opportunity",
]

RECEIPT_KEYWORDS = [
    "total", "subtotal", "tax", "payment", "paid", "cash", "credit",
    "debit", "visa", "mastercard", "transaction", "receipt", "purchase",
    "order", "item", "price", "amount", "balance", "refund", "change",
    "invoice", "billing", "charge", "fee", "cost",
]

DEADLINE_KEYWORDS = [
    "deadline", "due", "by", "submit by", "no later than", "closes",
    "last day", "final date", "cutoff", "expires", "must be received",
    "applications close", "submission deadline",
]

# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at module load for performance.
# ---------------------------------------------------------------------------

# Matches common date formats: "March 15", "03/15/2024", "2024-03-15", etc.
_DATE_PATTERN = re.compile(
    r"""
    \b(?:
        (?:January|February|March|April|May|June|July|August|
           September|October|November|December)
        \s+\d{1,2}(?:,\s*\d{4})?
    |
        \d{1,2}/\d{1,2}/\d{2,4}
    |
        \d{4}-\d{2}-\d{2}
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches email addresses.
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Matches common US phone formats: (404) 555-0192, 404-555-0192, 4045550192
_PHONE_PATTERN = re.compile(
    r"""
    \b(?:
        \(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}
    )\b
    """,
    re.VERBOSE,
)

# Matches dollar amounts: $5,000  $89.99  $1,200.00
_DOLLAR_PATTERN = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
)

# Sentence boundary: split on . ! ? followed by whitespace or end of string.
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def character_count(text: str) -> int:
    """Total number of characters in the text."""
    return len(text)


def word_count(text: str) -> int:
    """Number of whitespace-delimited tokens."""
    return len(text.split())


def sentence_count(text: str) -> int:
    """Approximate sentence count based on punctuation boundaries."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return max(1, len(sentences))


def date_count(text: str) -> int:
    """Number of date-like strings found in the text."""
    return len(_DATE_PATTERN.findall(text))


def email_count(text: str) -> int:
    """Number of email addresses found in the text."""
    return len(_EMAIL_PATTERN.findall(text))


def phone_count(text: str) -> int:
    """Number of phone number patterns found in the text."""
    return len(_PHONE_PATTERN.findall(text))


def dollar_amount_count(text: str) -> int:
    """Number of dollar amount patterns found in the text."""
    return len(_DOLLAR_PATTERN.findall(text))


def _count_keywords(text: str, keywords: list[str]) -> int:
    """Count how many distinct keywords from the list appear in the text."""
    lower = text.lower()
    return sum(1 for kw in keywords if kw in lower)


def action_word_count(text: str) -> int:
    """Number of action words detected (submit, apply, attend, etc.)."""
    return _count_keywords(text, ACTION_WORDS)


def education_keyword_count(text: str) -> int:
    """Number of education-related keywords detected."""
    return _count_keywords(text, EDUCATION_KEYWORDS)


def job_keyword_count(text: str) -> int:
    """Number of job/career-related keywords detected."""
    return _count_keywords(text, JOB_KEYWORDS)


def receipt_keyword_count(text: str) -> int:
    """Number of receipt/transaction-related keywords detected."""
    return _count_keywords(text, RECEIPT_KEYWORDS)


def deadline_keyword_count(text: str) -> int:
    """Number of deadline-related phrases detected."""
    return _count_keywords(text, DEADLINE_KEYWORDS)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

FEATURE_ORDER = [
    "character_count",
    "word_count",
    "sentence_count",
    "date_count",
    "email_count",
    "phone_count",
    "dollar_amount_count",
    "action_word_count",
    "education_keyword_count",
    "job_keyword_count",
    "receipt_keyword_count",
    "deadline_keyword_count",
]


def extract_features(text: str) -> dict[str, Union[int, float]]:
    """
    Extract all numeric features from a single document string.

    Returns a dict keyed by feature name in the canonical FEATURE_ORDER.
    Use extract_feature_vector() to get a plain list in the correct order.
    """
    return {
        "character_count": character_count(text),
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "date_count": date_count(text),
        "email_count": email_count(text),
        "phone_count": phone_count(text),
        "dollar_amount_count": dollar_amount_count(text),
        "action_word_count": action_word_count(text),
        "education_keyword_count": education_keyword_count(text),
        "job_keyword_count": job_keyword_count(text),
        "receipt_keyword_count": receipt_keyword_count(text),
        "deadline_keyword_count": deadline_keyword_count(text),
    }


def extract_feature_vector(text: str) -> list[float]:
    """
    Return features as a list of floats in FEATURE_ORDER.

    This is the format expected by the PyTorch model and Core ML model.
    The Swift app must produce values in the same order.
    """
    feats = extract_features(text)
    return [float(feats[name]) for name in FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Also expose extraction of text entities (used by AnalyzeView in the app)
# ---------------------------------------------------------------------------

def extract_dates(text: str) -> list[str]:
    """Return all date strings found in the text."""
    return _DATE_PATTERN.findall(text)


def extract_emails(text: str) -> list[str]:
    """Return all email addresses found in the text."""
    return _EMAIL_PATTERN.findall(text)


def extract_phones(text: str) -> list[str]:
    """Return all phone numbers found in the text."""
    return _PHONE_PATTERN.findall(text)


def extract_dollar_amounts(text: str) -> list[str]:
    """Return all dollar amount strings found in the text."""
    return _DOLLAR_PATTERN.findall(text)


def extract_action_words(text: str) -> list[str]:
    """Return which action words from the ACTION_WORDS list are present."""
    lower = text.lower()
    return [w for w in ACTION_WORDS if w in lower]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Applications are due March 15. Submit your resume, transcript, and "
        "recommendation letter to scholarships@gsu.edu. The award is worth "
        "$5,000 per year. Contact us at (404) 555-0192 for questions."
    )
    print("Sample text:")
    print(f"  {sample}\n")

    feats = extract_features(sample)
    print("Extracted features:")
    for name in FEATURE_ORDER:
        print(f"  {name:<30} {feats[name]}")

    print("\nExtracted entities:")
    print(f"  dates:   {extract_dates(sample)}")
    print(f"  emails:  {extract_emails(sample)}")
    print(f"  phones:  {extract_phones(sample)}")
    print(f"  dollars: {extract_dollar_amounts(sample)}")
    print(f"  actions: {extract_action_words(sample)}")
