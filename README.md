# Citation-Aware Research Assistant

A Python tool for asking questions about scientific papers and getting citation-backed answers. Built for Google Colab with Groq API.

## Features

- **Citation-aware chunking** - Preserves citation relationships
- **Section-aware splitting** - Maintains document structure context
- **Semantic clustering** - Groups related content intelligently
- **HyDE retrieval** - Enhanced query matching
- **Interactive interface** - Simple Q&A in Colab

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/username/research-assistant.git
   ```

2. **Open in Colab**
   - Upload `src/research_assistant_colab.py` to Google Colab
   - Install dependencies:
   ```python
   !pip install PyPDF2 langchain sentence-transformers faiss-cpu requests ipywidgets pdfplumber scikit-learn
   ```

3. **Get Groq API key**
   - Sign up at [Groq Console](https://console.groq.com)
   - Copy your API key

4. **Add PDFs**
   - Upload PDFs to Colab's `./pdfs/` directory

5. **Run and ask questions**
   ```
   Query: What are the key findings on GNN-Ret?
   Answer: GNN-Ret enhances retrieval for QA by exploiting passage relatedness, outperforming SBERT on datasets like Quality (Section 3.2).
   ```

## Requirements

- Python 3.8+
- Groq API key
- Google Colab (recommended)

## Structure

```
research-assistant/
├── src/
│   ├── research_assistant_colab.py
│   └── requirements.txt
├── data/sample_pdfs/
├── results/
└── docs/
```


