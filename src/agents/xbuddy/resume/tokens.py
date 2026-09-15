"""Token counting and budget splitting, measured with the embedding model's tokenizer.

`cl100k_base` is the tokenizer of `text-embedding-3-small`, so a chunk's
`token_count` here is what the embedding call will actually be charged for and
limited by — not an approximation from word counts.
"""

import re
from functools import cache

import tiktoken

ENCODING_NAME = "cl100k_base"

# One unit of splitting: a run of non-whitespace plus the whitespace after it, so
# joining units reproduces the text exactly, newlines included.
_WORD = re.compile(r"\S+\s*")


@cache
def _encoding() -> tiktoken.Encoding:
    # First use in a fresh environment downloads the BPE file (~1.7 MB) and caches
    # it; the Docker image will need that done at build time rather than on the
    # first upload. Stage 4 concern, noted here where the dependency lives.
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _hard_split_word(word: str, max_tokens: int) -> list[str]:
    """Split one pathological unit (a 500-token URL, a base64 blob) by token ids.

    Only reached when a single run of non-whitespace exceeds the whole budget.
    Decoding a token slice can land mid-character; tiktoken substitutes rather than
    raising, which is acceptable for input this malformed.
    """
    ids = _encoding().encode(word)
    return [_encoding().decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def _tail(text: str, overlap_tokens: int) -> str:
    """A suffix of `text` holding at least `overlap_tokens`, for the next piece to repeat.

    Line-aligned when whole trailing lines reach the budget within twice its size,
    so the next piece opens on a complete bullet rather than mid-sentence.
    Word-aligned otherwise: a single line far longer than the overlap would make the
    carry crowd out the new content it exists to connect.
    """
    lines = text.rstrip("\n").split("\n")
    taken_lines: list[str] = []
    for line in reversed(lines):
        taken_lines.insert(0, line)
        if count_tokens("\n".join(taken_lines)) >= overlap_tokens:
            break
    line_tail = "\n".join(taken_lines)
    if len(taken_lines) < len(lines) and count_tokens(line_tail) <= 2 * overlap_tokens:
        return line_tail + "\n"

    words = _WORD.findall(text)
    taken: list[str] = []
    for word in reversed(words):
        taken.insert(0, word)
        if count_tokens("".join(taken)) >= overlap_tokens:
            break
    return "".join(taken)


def split_to_budget(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split `text` into pieces of at most `max_tokens`, each overlapping the last.

    Pieces end on a line boundary when one is close enough — a bullet is a better
    place to cut than the middle of a sentence — and on a word boundary otherwise.
    Each piece after the first begins with roughly `overlap_tokens` from the end of
    the previous one, so a fact straddling a cut survives whole in one of them.

    Text already within budget comes back as a single piece, unchanged.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    units: list[str] = []
    for word in _WORD.findall(text):
        if count_tokens(word) > max_tokens:
            units.extend(_hard_split_word(word, max_tokens))
        else:
            units.append(word)

    pieces: list[str] = []
    carry = ""
    start = 0
    while start < len(units):
        piece = carry
        consumed = start
        while consumed < len(units) and count_tokens(piece + units[consumed]) <= max_tokens:
            piece += units[consumed]
            consumed += 1

        if consumed == start:
            # Not even one new unit fits beside the carried overlap. Drop the carry
            # rather than loop forever; the unit alone always fits.
            piece = units[start]
            consumed = start + 1
        elif consumed < len(units):
            # Prefer to end at the last newline, provided that still leaves at least
            # half a piece of new content — otherwise a cut mid-line is the lesser
            # evil than a piece made mostly of overlap.
            new_part = "".join(units[start:consumed])
            newline = new_part.rfind("\n")
            if newline != -1 and count_tokens(new_part[: newline + 1]) >= max_tokens // 2:
                kept = new_part[: newline + 1]
                # Recount how many whole units that covers.
                covered, length = start, 0
                while covered < consumed and length + len(units[covered]) <= len(kept):
                    length += len(units[covered])
                    covered += 1
                if covered > start:
                    piece = carry + "".join(units[start:covered])
                    consumed = covered

        pieces.append(piece.strip())
        if consumed >= len(units):
            break
        carry = _tail(piece, overlap_tokens)
        start = consumed

    return [p for p in pieces if p]
