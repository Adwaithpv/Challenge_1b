# JSON Post-Processing Scripts

This directory contains scripts to ensure that the output JSON from the PDF analysis is properly formatted according to the required schema with proper indentation and structure.

## Files

- `post_process_output.py` - Main post-processing script for formatting and validating JSON output
- `run_with_post_processing.py` - Wrapper script that runs the main solution and automatically post-processes the output
- `POST_PROCESSING_README.md` - This documentation file

## Required JSON Schema

The post-processing scripts ensure the output JSON follows this exact schema:

```json
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
```

## Usage

### 1. Post-Process Existing JSON File

To format an existing JSON output file:

```bash
python post_process_output.py input.json -o formatted_output.json
```

Options:
- `-o, --output`: Specify output file (default: print to stdout)
- `--indent`: Number of spaces for indentation (default: 2)
- `--validate-only`: Only validate the JSON structure without processing

Examples:

```bash
# Format and save to file
python post_process_output.py output.json -o formatted_output.json

# Format and print to console
python post_process_output.py output.json

# Validate only (no processing)
python post_process_output.py output.json --validate-only

# Custom indentation
python post_process_output.py output.json -o output.json --indent 4
```

### 2. Run Solution with Automatic Post-Processing

To run the main solution and automatically post-process the output:

```bash
python run_with_post_processing.py input.pdf [output.json]
```

Examples:

```bash
# Process PDF and print formatted output to console
python run_with_post_processing.py document.pdf

# Process PDF and save formatted output to file
python run_with_post_processing.py document.pdf output.json
```

## Features

### Automatic Field Mapping

The post-processing script automatically handles different field names that might be present in the raw output:

**Title fields:**
- `title`

**Outline fields:**
- `outline`
- `headings`
- `toc`
- `table_of_contents`

**Level fields:**
- `level`
- `heading_level`
- `type`
- `indentation`

**Text fields:**
- `text`
- `label`
- `content`
- `title`
- `heading`

**Page fields:**
- `page`
- `page_number`
- `page_num`
- `pageNumber`

### Data Type Conversion

The script automatically converts data types to match the required schema:

- **Level**: Converts to proper heading format (H1, H2, H3, etc.)
- **Text**: Ensures string type and strips whitespace
- **Page**: Converts to integer type

### Validation

The script validates the output against the required schema and reports any errors:

- Missing required fields
- Incorrect data types
- Invalid structure

### Error Handling

- Graceful handling of malformed JSON
- Automatic field mapping for different output formats
- Default values for missing data
- Detailed error reporting

## Integration with Main Solution

The post-processing can be integrated into the main solution by modifying `round1a_solution.py` to use the `JSONPostProcessor` class:

```python
from post_process_output import JSONPostProcessor

# After generating the result
processor = JSONPostProcessor()
formatted_result = processor.validate_and_fix_structure(result)
json_output = processor.format_json(formatted_result, indent=2)
```

## Example Output

**Before post-processing:**
```json
{"title": "", "headings": [{"level": "H1", "text": "FORMULA", "page_number": 1}, {"level": "H1", "text": "FOOTNOTE", "page_number": 1}], "total_headings": 2, "max_heading_level": 1, "processing_time_seconds": 0.01}
```

**After post-processing:**
```json
{
  "title": "",
  "outline": [
    {
      "level": "H1",
      "text": "FORMULA",
      "page": 1
    },
    {
      "level": "H1",
      "text": "FOOTNOTE",
      "page": 1
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Missing required fields**: The script will attempt to map from alternative field names
2. **Invalid data types**: The script will convert data types automatically
3. **Malformed JSON**: The script will report JSON parsing errors

### Error Messages

- `"Missing required field: 'title'"` - Title field not found
- `"Missing required field: 'outline'"` - Outline field not found
- `"Outline item X missing required field: 'level'"` - Missing level in outline item
- `"Invalid JSON in file: ..."` - JSON parsing error

## Performance

The post-processing scripts are designed to be lightweight and fast:
- Minimal memory usage
- Fast field mapping and validation
- Efficient JSON formatting
- No external dependencies beyond Python standard library 