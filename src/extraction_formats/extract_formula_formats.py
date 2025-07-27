import io
from typing import Any
from PIL.Image import Image

# Try to import optional dependency
try:
    from rapid_latex_ocr import LaTeXOCR
    RAPID_LATEX_OCR_AVAILABLE = True
except ImportError:
    LaTeXOCR = Any  # Fallback type when import fails
    RAPID_LATEX_OCR_AVAILABLE = False
    print("Warning: rapid_latex_ocr not available. Formula extraction will be disabled.")

from data_model.PdfImages import PdfImages
from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType


def has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" for char in text)


def get_latex_format(model: LaTeXOCR, formula_image: Image):
    if not RAPID_LATEX_OCR_AVAILABLE or model is None:
        return "[Formula extraction not available - rapid_latex_ocr not installed]"
        
    buffer = io.BytesIO()
    formula_image.save(buffer, format="jpeg")
    image_bytes = buffer.getvalue()
    result, elapsed_time = model(image_bytes)
    return result


def extract_formula_format(pdf_images: PdfImages, predicted_segments: list[PdfSegment]):
    if not RAPID_LATEX_OCR_AVAILABLE:
        print("Warning: Formula extraction skipped - rapid_latex_ocr not available")
        return
        
    formula_segments = [
        (index, segment) for index, segment in enumerate(predicted_segments) if segment.segment_type == TokenType.FORMULA
    ]
    if not formula_segments:
        return

    try:
        model = LaTeXOCR()
    except Exception:
        print("Warning: Could not initialize LaTeXOCR model")
        return

    for index, formula_segment in formula_segments:
        page_image: Image = pdf_images.pdf_images[formula_segment.page_number - 1]
        left, top = formula_segment.bounding_box.left, formula_segment.bounding_box.top
        width, height = formula_segment.bounding_box.width, formula_segment.bounding_box.height
        formula_image = page_image.crop((left, top, left + width, top + height))
        extracted_formula = get_latex_format(model, formula_image)
        if has_arabic(extracted_formula):
            continue
        predicted_segments[index].text_content = extracted_formula
