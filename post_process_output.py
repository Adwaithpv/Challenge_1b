#!/usr/bin/env python3
"""
Post-processing script for Round 1A output JSON formatting.

This script ensures that the output JSON from the PDF analysis is properly formatted
according to the required schema with proper indentation and structure.

Required JSON Schema:
{
  "title": "string",
  "outline": [
    {
      "level": "string",
      "text": "string", 
      "page": "integer"
    }
  ]
}
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union


class JSONPostProcessor:
    """Post-processor for formatting and validating JSON output."""
    
    def __init__(self):
        self.required_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "outline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "string"},
                            "text": {"type": "string"},
                            "page": {"type": "integer"}
                        },
                        "required": ["level", "text", "page"]
                    }
                }
            },
            "required": ["title", "outline"]
        }
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def validate_and_fix_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix the JSON structure to match required schema.
        
        Args:
            data: Input JSON data
            
        Returns:
            Properly formatted JSON data
        """
        result = {
            "title": "",
            "outline": []
        }
        
        # Extract title
        if "title" in data:
            result["title"] = str(data["title"]).strip()
        
        # Extract outline from various possible field names
        outline_data = []
        
        # Check for different possible field names
        outline_fields = ["outline", "headings", "toc", "table_of_contents"]
        for field in outline_fields:
            if field in data and isinstance(data[field], list):
                outline_data = data[field]
                break
        
        # Process outline items
        for item in outline_data:
            if not isinstance(item, dict):
                continue
                
            # Extract level
            level = self._extract_level(item)
            
            # Extract text
            text = self._extract_text(item)
            
            # Extract page
            page = self._extract_page(item)
            
            # Only add if we have valid data
            if level and text and page is not None:
                result["outline"].append({
                    "level": level,
                    "text": text,
                    "page": page
                })
        
        return result
    
    def _extract_level(self, item: Dict[str, Any]) -> str:
        """Extract heading level from item."""
        # Check various possible field names for level
        level_fields = ["level", "heading_level", "type", "indentation"]
        
        for field in level_fields:
            if field in item:
                level = str(item[field]).strip()
                if level:
                    # Ensure it's in the correct format (H1, H2, etc.)
                    if level.upper().startswith('H'):
                        return level.upper()
                    elif level.isdigit():
                        return f"H{level}"
                    else:
                        # Try to convert other formats
                        try:
                            level_num = int(level)
                            return f"H{level_num}"
                        except (ValueError, TypeError):
                            return level.upper()
        
        return "H1"  # Default level
    
    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text content from item."""
        # Check various possible field names for text
        text_fields = ["text", "label", "content", "title", "heading"]
        
        for field in text_fields:
            if field in item:
                text = str(item[field]).strip()
                if text:
                    return text
        
        return ""
    
    def _extract_page(self, item: Dict[str, Any]) -> Union[int, None]:
        """Extract page number from item."""
        # Check various possible field names for page
        page_fields = ["page", "page_number", "page_num", "pageNumber"]
        
        for field in page_fields:
            if field in item:
                try:
                    page = item[field]
                    if isinstance(page, (int, float)):
                        return int(page)
                    elif isinstance(page, str) and page.isdigit():
                        return int(page)
                except (ValueError, TypeError):
                    continue
        
        return 1  # Default page number
    
    def format_json(self, data: Dict[str, Any], indent: int = 2) -> str:
        """
        Format JSON with proper indentation.
        
        Args:
            data: JSON data to format
            indent: Number of spaces for indentation
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(data, indent=indent, ensure_ascii=False, separators=(',', ': '))
    
    def validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data against required schema.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if "title" not in data:
            errors.append("Missing required field: 'title'")
        
        if "outline" not in data:
            errors.append("Missing required field: 'outline'")
            return errors
        
        # Validate title
        if not isinstance(data["title"], str):
            errors.append("'title' must be a string")
        
        # Validate outline
        if not isinstance(data["outline"], list):
            errors.append("'outline' must be an array")
            return errors
        
        # Validate outline items
        for i, item in enumerate(data["outline"]):
            if not isinstance(item, dict):
                errors.append(f"Outline item {i} must be an object")
                continue
            
            # Check required fields in outline item
            for field in ["level", "text", "page"]:
                if field not in item:
                    errors.append(f"Outline item {i} missing required field: '{field}'")
            
            # Validate field types
            if "level" in item and not isinstance(item["level"], str):
                errors.append(f"Outline item {i} 'level' must be a string")
            
            if "text" in item and not isinstance(item["text"], str):
                errors.append(f"Outline item {i} 'text' must be a string")
            
            if "page" in item and not isinstance(item["page"], int):
                errors.append(f"Outline item {i} 'page' must be an integer")
        
        return errors
    
    def process_file(self, input_path: str, output_path: str = None, indent: int = 2) -> str:
        """
        Process a JSON file and format it according to the required schema.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file (optional)
            indent: Number of spaces for indentation
            
        Returns:
            Formatted JSON string
        """
        # Load and process the data
        data = self.load_json(input_path)
        processed_data = self.validate_and_fix_structure(data)
        
        # Validate the processed data
        errors = self.validate_schema(processed_data)
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"  - {error}")
            print("Attempting to fix structure...")
        
        # Format the JSON
        formatted_json = self.format_json(processed_data, indent)
        
        # Save to output file if specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"Formatted JSON saved to: {output_path}")
        
        return formatted_json


def main():
    """Main entry point for the post-processing script."""
    parser = argparse.ArgumentParser(
        description="Post-process JSON output to ensure proper formatting and schema compliance"
    )
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument(
        "-o", "--output", 
        help="Output JSON file (default: print to stdout)"
    )
    parser.add_argument(
        "--indent", 
        type=int, 
        default=2,
        help="Number of spaces for indentation (default: 2)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the JSON structure without processing"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        processor = JSONPostProcessor()
        
        if args.validate_only:
            # Only validate
            data = processor.load_json(str(input_path))
            errors = processor.validate_schema(data)
            if errors:
                print("Validation errors:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("JSON structure is valid!")
        else:
            # Process and format
            formatted_json = processor.process_file(
                str(input_path), 
                args.output, 
                args.indent
            )
            
            if not args.output:
                print(formatted_json)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 