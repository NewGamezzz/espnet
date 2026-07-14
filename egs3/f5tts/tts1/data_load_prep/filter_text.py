#!/usr/bin/env python3
"""
Filter text utterances based on repetitive patterns and long digit sequences.

This script detects and filters out utterances containing:
1. Excessive character repetition (e.g., "Ooooooooo...")
2. Excessive word/phrase repetition (e.g., "uh, uh, uh...")
3. High ratio of repetitive content
4. Long sequences of digits (e.g., "1234567890")
5. Wrong-language characters (e.g., Arabic/Japanese in EN text)
6. N-gram repetition (ported from F5-TTS repetition_found logic)

Usage:
    python filter_text.py "text to check" [options]
    echo "text to check" | python filter_text.py [options]

Exit codes:
    0: Text is acceptable (should be kept)
    1: Text should be filtered out
    2: Error in processing
"""

import argparse
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Language-specific character filters (from F5-TTS prepare_emilia_v2.py)
# Characters that should not appear in EN transcriptions
# ---------------------------------------------------------------------------
LANG_CHAR_FILTERS = {
    "EN": ["ا", "い", "て"],   # Arabic / Japanese chars
    "ZH": [],                   # No cross-language filter needed for ZH
}


def detect_char_repetition(text, max_repeat=10):
    """
    Detect excessive character repetition.

    Args:
        text: Input text
        max_repeat: Maximum allowed consecutive character repetitions

    Returns:
        True if excessive repetition detected, False otherwise
    """
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return bool(re.search(pattern, text, re.IGNORECASE))


def detect_word_repetition(text, max_repeat=5, max_word_len=3):
    """
    Detect excessive word/phrase repetition of short filler words.

    Args:
        text: Input text
        max_repeat: Maximum allowed consecutive word repetitions
        max_word_len: Maximum word length to consider for repetition detection

    Returns:
        True if excessive repetition detected, False otherwise
    """
    words = re.split(r"[\s,;.!?]+", text.lower())
    words = [w for w in words if w]

    if len(words) < max_repeat:
        return False

    consecutive_count = 1
    prev_word = None

    for word in words:
        if len(word) <= max_word_len:
            if word == prev_word:
                consecutive_count += 1
                if consecutive_count > max_repeat:
                    return True
            else:
                consecutive_count = 1
                prev_word = word
        else:
            consecutive_count = 1
            prev_word = None

    return False


def detect_ngram_repetition(text, length=4):
    """
    Detect repeated n-grams in text, ported from F5-TTS repetition_found().

    Splits on whitespace and checks whether any n-gram of `length` words
    appears more than once.

    Args:
        text: Input text
        length: N-gram length to check

    Returns:
        True if a repeated n-gram is found, False otherwise
    """
    words = text.split()
    if len(words) < length:
        return False

    ngrams = [tuple(words[i:i + length]) for i in range(len(words) - length + 1)]
    seen = set()
    for ng in ngrams:
        if ng in seen:
            return True
        seen.add(ng)
    return False


def detect_repetition_ratio(text, max_ratio=0.5):
    """
    Detect if the text has too high a ratio of repetitive content.

    Args:
        text: Input text
        max_ratio: Maximum allowed ratio of repetitive characters

    Returns:
        True if repetition ratio exceeds threshold, False otherwise
    """
    if not text:
        return True

    text_clean = re.sub(r"\s+", "", text.lower())
    if not text_clean:
        return True

    char_counts = Counter(text_clean)
    most_common_count = char_counts.most_common(1)[0][1]
    return (most_common_count / len(text_clean)) > max_ratio


def detect_long_digits(text, threshold=15):
    """
    Detect lines containing long digit runs.
    Matches raw digits or digits separated by common punctuation.
    """
    if threshold <= 0:
        return False

    patt = re.compile(r"\d{" + str(threshold) + r",}")
    if patt.search(text):
        return True

    clean = text.replace(",", "").replace("|", "").replace("'", "").replace(".", "")
    return bool(patt.search(clean))


def detect_wrong_language_chars(text, lang):
    """
    Detect characters that should not appear for a given language.
    Based on en_filters logic from F5-TTS prepare_emilia_v2.py.

    Args:
        text: Input text
        lang: Language code (e.g. "EN", "ZH")

    Returns:
        True if a disallowed character is found, False otherwise
    """
    filters = LANG_CHAR_FILTERS.get(lang.upper(), [])
    return any(ch in text for ch in filters)


def should_filter(
    text,
    lang="EN",
    max_char_repeat=10,
    max_word_repeat=5,
    max_repeat_ratio=0.5,
    max_long_digits=15,
    ngram_length=4,
    verbose=False,
):
    """
    Determine if text should be filtered out.

    Args:
        text: Input text to check
        lang: Language code for language-specific char filters
        max_char_repeat: Maximum consecutive character repetitions
        max_word_repeat: Maximum consecutive word repetitions
        max_repeat_ratio: Maximum ratio of repetitive characters
        max_long_digits: Maximum length of digit sequences
        ngram_length: N-gram length for repetition_found check
        verbose: Print reason for filtering

    Returns:
        True if text should be filtered, False if it should be kept
    """
    if not text or not text.strip():
        if verbose:
            print("FILTER: Empty text", file=sys.stderr)
        return True

    if detect_wrong_language_chars(text, lang):
        if verbose:
            print(f"FILTER: Wrong-language characters detected for lang={lang}", file=sys.stderr)
        return True

    if detect_char_repetition(text, max_char_repeat):
        if verbose:
            print(f"FILTER: Excessive character repetition (>{max_char_repeat})", file=sys.stderr)
        return True

    if detect_word_repetition(text, max_word_repeat):
        if verbose:
            print(f"FILTER: Excessive word repetition (>{max_word_repeat})", file=sys.stderr)
        return True

    if detect_ngram_repetition(text, ngram_length):
        if verbose:
            print(f"FILTER: Repeated {ngram_length}-gram found", file=sys.stderr)
        return True

    if detect_repetition_ratio(text, max_repeat_ratio):
        if verbose:
            print(f"FILTER: High repetition ratio (>{max_repeat_ratio})", file=sys.stderr)
        return True

    if detect_long_digits(text, max_long_digits):
        if verbose:
            print(f"FILTER: Long digit sequence (>{max_long_digits} digits)", file=sys.stderr)
        return True

    if verbose:
        print("KEEP: Text passes all filters", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter text based on repetitive patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter_text.py "This is normal text"
    echo "Ooooooooooooooooo" | python filter_text.py
    python filter_text.py "uh uh uh" --max_word_repeat 2
    python filter_text.py "hello hello hello hello" --ngram_length 2
    python filter_text.py "some text" --lang ZH

Exit codes:
    0: Text should be kept
    1: Text should be filtered out
    2: Error
        """,
    )

    parser.add_argument("text", nargs="?", help="Text to check (if not provided, reads from stdin)")
    parser.add_argument("--lang", type=str, default="EN", help="Language code for character filters (default: EN)")
    parser.add_argument("--max_char_repeat", type=int, default=10)
    parser.add_argument("--max_word_repeat", type=int, default=5)
    parser.add_argument("--max_repeat_ratio", type=float, default=0.5)
    parser.add_argument("--max_long_digits", type=int, default=15)
    parser.add_argument("--ngram_length", type=int, default=4,
                        help="N-gram length for repetition check (default: 4, from F5-TTS)")
    parser.add_argument("--verbose", action="store_true", help="Print filtering reason to stderr")

    args = parser.parse_args()

    text = args.text if args.text else sys.stdin.read().strip()

    try:
        should_be_filtered = should_filter(
            text,
            lang=args.lang,
            max_char_repeat=args.max_char_repeat,
            max_word_repeat=args.max_word_repeat,
            max_repeat_ratio=args.max_repeat_ratio,
            max_long_digits=args.max_long_digits,
            ngram_length=args.ngram_length,
            verbose=args.verbose,
        )
        sys.exit(1 if should_be_filtered else 0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()