"""Build small, real PDFs for the resume extraction tests. No new dependency.

A hand-written writer: Helvetica with WinAnsiEncoding, one text line per `Tm`, and
a correct cross-reference table. Deterministic, so the same text always makes the
same bytes, and small enough to read.

Long lines are **word-wrapped** the way a real PDF lays them out. That is the point:
a resume's bullets come out of pypdf as several physical lines, and the chunker has
to put them back together. A writer that emitted each bullet on one line would hide
exactly the case that breaks entry splitting on real files.

Imported directly by the tests in this directory — pytest puts a test file's own
directory on sys.path, and `tests/` is not a package.
"""

import io
import textwrap

from pypdf import PdfReader, PdfWriter

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
MARGIN_X, TOP, BOTTOM = 50, 750, 50
FONT_SIZE, LEADING = 9, 12
WRAP_CHARS = 95  # comfortably inside the page at 9pt Helvetica


def _escape(line: str) -> str:
    out = []
    for byte in line.encode("cp1252"):
        if byte in b"()\\":
            out.append("\\" + chr(byte))
        elif byte < 32 or byte > 126:
            out.append(f"\\{byte:03o}")
        else:
            out.append(chr(byte))
    return "".join(out)


def _wrap(text: str) -> list[str]:
    physical: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            physical.append("")
            continue
        physical.extend(textwrap.wrap(line, WRAP_CHARS, break_on_hyphens=False) or [""])
    return physical


def _paginate(lines: list[str]) -> list[list[str]]:
    per_page = (TOP - BOTTOM) // LEADING
    return [lines[i : i + per_page] for i in range(0, len(lines), per_page)] or [[]]


def _content_stream(lines: list[str]) -> bytes:
    ops = ["BT", f"/F1 {FONT_SIZE} Tf"]
    y = TOP
    for line in lines:
        if line:
            ops.append(f"1 0 0 1 {MARGIN_X} {y} Tm ({_escape(line)}) Tj")
        y -= LEADING
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


_IMAGE_STREAM = (
    b"q 200 0 0 100 50 600 cm /Im1 Do Q"
)
# A 2x2 RGB image: enough for a page whose only content is a picture.
_IMAGE_PIXELS = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])


def _assemble(page_streams: list[bytes], *, with_image: bool = False) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # placeholder, filled once the page tree exists
    pages_id = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    image = None
    if with_image:
        image = add(
            b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /Length %d >>\nstream\n" % len(_IMAGE_PIXELS)
            + _IMAGE_PIXELS
            + b"\nendstream"
        )

    page_ids = []
    for stream in page_streams:
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        resources = b"<< /Font << /F1 %d 0 R >>" % font
        if image:
            resources += b" /XObject << /Im1 %d 0 R >>" % image
        resources += b" >>"
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] /Resources "
                % (pages_id, PAGE_WIDTH, PAGE_HEIGHT)
                + resources
                + b" /Contents %d 0 R >>" % content
            )
        )

    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref))
    return out.getvalue()


def text_pdf(text: str) -> bytes:
    """A text PDF of `text`, wrapped and paginated like a real document."""
    return _assemble([_content_stream(page) for page in _paginate(_wrap(text))])


def pages_pdf(pages: list[str]) -> bytes:
    """One PDF page per string, each wrapped but never paginated further."""
    return _assemble([_content_stream(_wrap(page)) for page in pages])


def image_only_pdf(page_count: int = 1) -> bytes:
    """Pages whose only content is an image — what a scanned resume looks like."""
    return _assemble([_IMAGE_STREAM] * page_count, with_image=True)


def encrypted_pdf(text: str, *, user_password: str, owner_password: str = "owner") -> bytes:
    """`text_pdf(text)` re-saved with AES-256 encryption.

    An empty `user_password` gives the common "restrictions only" case: encrypted,
    yet openable by anyone without a password.
    """
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(text_pdf(text))))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="AES-256")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
