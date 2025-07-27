
# Persona-Driven Document Intelligence

This project is a sophisticated document analysis system that acts as an intelligent document analyst. It is designed to ingest a collection of PDF documents, a user persona, and a job-to-be-done, and then output a ranked list of the most relevant document sections that address the user's specific needs.

## Key Features

- **Persona-Driven Analysis**: The system tailors its analysis to specific user roles and objectives, ensuring that the results are highly relevant to the user's context.
- **Semantic Understanding**: It uses state-of-the-art sentence-transformer embeddings (`all-MiniLM-L6-v2`) to perform a meaning-based ranking of document sections, going beyond simple keyword matching.
- **Content Hydration**: The system intelligently extracts the full text of document sections by leveraging a pre-existing structural analysis, which defines the boundaries of titles, headings, and other elements.
- **Extractive Summarization**: It generates concise, extractive summaries of the most relevant sections using a BERT-based model, allowing users to quickly grasp the key takeaways.
- **Optimized Performance**: The solution is engineered for efficiency, with model reuse and optimized algorithms that ensure processing completes in under 60 seconds.
- **Offline Execution**: The system is fully self-contained and can run without an internet connection, as all required models are pre-downloaded and cached.

## Architecture

The system follows a core pipeline:

`Round 1A JSON + PDFs → Content Hydration → Semantic Vectorization → Relevance Ranking → Text Summarization → Ranked Output`

1.  **Content Hydration**: The system loads the structural metadata from the Round 1A JSON files and uses it to extract the full text of each section from the corresponding PDFs. This creates semantically coherent content blocks for analysis.
2.  **Semantic Vectorization**: The user persona and job-to-be-done are combined to form a descriptive search query. The `all-MiniLM-L6-v2` model is then used to create 384-dimensional vector embeddings for both the query and the content of each document section.
3.  **Relevance Ranking**: The system calculates the cosine similarity between the query vector and each section vector to determine their semantic relevance. All sections are then ranked globally based on their similarity scores.
4.  **Text Summarization**: The top-ranked sections are processed by a BERT-based extractive summarizer, which uses the same `all-MiniLM-L6-v2` model to generate concise, 3-5 sentence summaries.

## Docker Execution Instructions

The recommended way to run this solution is by using the provided Docker container. This ensures a consistent and reproducible environment with all dependencies pre-installed.

### Prerequisites

-   Docker must be installed and running on your system.

### Building the Docker Image

First, build the Docker image using the following command from the root of the project directory:

```bash
docker build --platform linux/amd64 -t round1b-solution .
```

This command will create a Docker image named `round1b-solution` and pre-download the necessary models into the image, allowing for offline execution.

### Preparing the Input Directory

The Docker container expects the input files to be mounted to the `/app/input` directory. You will need to create an `input` directory and populate it with the required files:

-   A `PDFs` subdirectory containing the PDF documents to be analyzed.
-   The corresponding `.json` files from the Round 1A analysis, located in the root of the `input` directory.
-   An `input.json` file that specifies the user persona, job-to-be-done, and the list of documents to process, also in the root of the `input` directory.

The structure of the `input` directory should be as follows:

```
input/
├── PDFs/
│   ├── document1.pdf
│   └── document2.pdf
└── input.json
```

The `input.json` file should follow this format:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1 Title"
        },
        {
            "filename": "document2.pdf",
            "title": "Document 2 Title"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

### Running the Docker Container

Once the input directory is prepared, you can run the solution using the following command:

```bash
docker run -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" round1b-solution
```

This command will:

-   Mount your local `input` directory to the `/app/input` directory inside the container.
-   Mount your local `output` directory to the `/app/output` directory inside the container.
-   Execute the main script, which will process the documents and generate the output.

The results will be saved in a file named `output.json` inside the `output` directory.

### Output

The output is a JSON file containing the ranked list of relevant sections and their summaries. The structure of the output is as follows:

```json
{
    "metadata": {
        "input_documents": [
            "document1.pdf",
            "document2.pdf"
        ],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
        "processing_timestamp": "2025-01-16T12:00:00.000000"
    },
    "extracted_sections": [
        {
            "document": "document1.pdf",
            "section_title": "Section Title",
            "importance_rank": 1,
            "page_number": 1
        }
    ],
    "subsection_analysis": [
        {
            "document": "document1.pdf",
            "refined_text": "This is a summary of the section...",
            "page_number": 1
        }
    ]
}
```

## Local Execution (Without Docker)

If you prefer to run the solution locally without Docker, you can do so by following these steps:

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Command-Line Usage

You can run the solution from the command line by providing the path to the input JSON file:

```bash
python round1b_solution.py --input-json input/input.json --output-file output/output.json
```

Alternatively, you can provide the persona and job-to-be-done as command-line arguments:

```bash
python round1b_solution.py \
  --input-dir input/ \
  --output-file output/output.json \
  --persona "Travel Planner" \
  --job "Plan a trip of 4 days for a group of 10 college friends."
``` 