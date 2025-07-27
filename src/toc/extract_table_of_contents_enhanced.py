import tempfile
import uuid
import re
from os.path import join
from pathlib import Path
from typing import AnyStr
from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from toc.TOCExtractor import TOCExtractor
from configuration import service_logger
from toc.PdfSegmentation import PdfSegmentation

# Original title types
TITLE_TYPES = {TokenType.TITLE, TokenType.SECTION_HEADER}
SKIP_TYPES = {TokenType.TITLE, TokenType.SECTION_HEADER, TokenType.PAGE_HEADER, TokenType.PICTURE}

# Enhanced title detection for Text segments
HEADING_INDICATORS = [
    'checklist', 'test', 'export', 'skill', 'ultimate', 'guide', 'introduction',
    'conclusion', 'summary', 'overview', 'tips', 'tricks', 'professional',
    'sharing', 'expert', 'welcome', 'scenario', 'question', 'challenge'
]

def get_file_path(file_name, extension):
    return join(tempfile.gettempdir(), file_name + "." + extension)

def pdf_content_to_pdf_path(file_content):
    file_id = str(uuid.uuid1())
    pdf_path = Path(get_file_path(file_id, "pdf"))
    pdf_path.write_bytes(file_content)
    return pdf_path

def is_likely_heading_text(text: str, segment_type: TokenType) -> bool:
    """
    Enhanced logic to identify potential headings in Text segments.
    
    Args:
        text: The text content of the segment
        segment_type: The type of the segment
        
    Returns:
        True if this text segment is likely a heading
    """
    if not text or len(text.strip()) == 0:
        return False
    
    text_clean = text.strip().lower()
    
    # Criteria 1: Contains heading indicators
    for indicator in HEADING_INDICATORS:
        if indicator in text_clean:
            # Additional checks for heading-like patterns
            if (len(text.strip()) < 200 and  # Not too long
                not text_clean.endswith('.') or  # Doesn't end with period (not a sentence)
                any(keyword in text_clean for keyword in ['checklist', 'test', 'expert', 'ultimate'])):
                return True
    
    # Criteria 2: Short, standalone text that looks like a title
    if (len(text.strip()) < 100 and 
        len(text.strip()) > 10 and
        not text_clean.endswith('.')):
        
        # Check if it has title-like capitalization or formatting
        words = text.strip().split()
        if len(words) <= 8:  # Short phrases
            # Check for title case or all caps patterns
            capitalized_words = sum(1 for word in words if word and word[0].isupper())
            if capitalized_words >= len(words) * 0.5:  # At least half the words are capitalized
                return True
    
    # Criteria 3: Specific patterns for our problematic documents
    title_patterns = [
        r'^export expert.*skills?$',
        r'^.*checklist$',
        r'^test.*skills?$',
        r'^ultimate.*guide$',
        r'^general.*checklist$',
        r'^.*sharing.*checklist$',
        r'^professional tip.*$',
        r'^conclusion$'
    ]
    
    for pattern in title_patterns:
        if re.match(pattern, text_clean):
            return True
    
    return False

def skip_name_of_the_document(pdf_segments: list[PdfSegment], title_segments: list[PdfSegment]):
    segments_to_remove = []
    last_segment = None
    for segment in pdf_segments:
        if segment.segment_type not in SKIP_TYPES:
            break
        if segment.segment_type == TokenType.PAGE_HEADER or segment.segment_type == TokenType.PICTURE:
            continue
        if not last_segment:
            last_segment = segment
        else:
            if segment.bounding_box.right < last_segment.bounding_box.left + last_segment.bounding_box.width * 0.66:
                break
            last_segment = segment
        if segment.segment_type in TITLE_TYPES:
            segments_to_remove.append(segment)
    for segment in segments_to_remove:
        title_segments.remove(segment)

def get_pdf_segments_from_segment_boxes_enhanced(pdf_features: PdfFeatures, segment_boxes: list[dict]) -> list[PdfSegment]:
    """Enhanced version that can identify more potential headings."""
    pdf_segments: list[PdfSegment] = []
    for segment_box in segment_boxes:
        left, top, width, height = segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        bounding_box = Rectangle.from_width_height(left, top, width, height)
        segment_type = TokenType.from_value(segment_box["type"])
        pdf_name = pdf_features.file_name
        segment = PdfSegment(segment_box["page_number"], bounding_box, segment_box["text"], segment_type, pdf_name)
        pdf_segments.append(segment)
    return pdf_segments

def extract_title_segments_enhanced(pdf_segments: list[PdfSegment]) -> list[PdfSegment]:
    """
    Enhanced title segment extraction that considers Text segments as potential headings.
    
    Args:
        pdf_segments: All PDF segments
        
    Returns:
        List of segments that are likely headings/titles
    """
    title_segments = []
    
    for segment in pdf_segments:
        # Original logic: include traditional title types
        if segment.segment_type in TITLE_TYPES:
            title_segments.append(segment)
        
        # Enhanced logic: check Text segments for heading patterns
        elif segment.segment_type == TokenType.TEXT:
            if is_likely_heading_text(segment.text_content, segment.segment_type):
                title_segments.append(segment)
                # Debug output to see what we're finding
                service_logger.info(f"Enhanced TOC: Found potential heading in Text segment: '{segment.text_content[:50]}...'")
    
    return title_segments

def extract_table_of_contents_enhanced(file: AnyStr, segment_boxes: list[dict], skip_document_name=False):
    """
    Enhanced TOC extraction that can recognize headings in Text segments.
    
    Args:
        file: PDF file content
        segment_boxes: Segment boxes from layout analysis
        skip_document_name: Whether to skip document name detection
        
    Returns:
        TOC data as list of dictionaries
    """
    service_logger.info("Getting TOC (Enhanced)")
    pdf_path = pdf_content_to_pdf_path(file)
    
    try:
        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path)
        pdf_segments: list[PdfSegment] = get_pdf_segments_from_segment_boxes_enhanced(pdf_features, segment_boxes)
        
        # Use enhanced title segment extraction
        title_segments = extract_title_segments_enhanced(pdf_segments)
        
        service_logger.info(f"Enhanced TOC: Found {len(title_segments)} potential title segments")
        
        if skip_document_name:
            skip_name_of_the_document(pdf_segments, title_segments)
        
        # If no title segments found with enhanced logic, fall back to original
        if not title_segments:
            service_logger.info("Enhanced TOC: No titles found with enhanced logic, falling back to original")
            title_segments = [segment for segment in pdf_segments if segment.segment_type in TITLE_TYPES]
        
        pdf_segmentation: PdfSegmentation = PdfSegmentation(pdf_features, title_segments)
        toc_instance: TOCExtractor = TOCExtractor(pdf_segmentation)
        
        return toc_instance.to_dict()
        
    finally:
        # Cleanup
        try:
            pdf_path.unlink()
        except FileNotFoundError:
            pass

# Keep the original function for compatibility
def get_pdf_segments_from_segment_boxes(pdf_features: PdfFeatures, segment_boxes: list[dict]) -> list[PdfSegment]:
    return get_pdf_segments_from_segment_boxes_enhanced(pdf_features, segment_boxes)

def extract_table_of_contents(file: AnyStr, segment_boxes: list[dict], skip_document_name=False):
    """Original function - redirects to enhanced version for better results."""
    return extract_table_of_contents_enhanced(file, segment_boxes, skip_document_name) 