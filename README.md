# Citation-Aware Research Assistant

A Python-based tool for question-answering on scientific papers, featuring citation-aware chunking, section-aware splitting, semantic chunking, HyDE-enhanced retrieval, and multi-hop retrieval, designed for Google Colab with Groq API integration.

# Project Overview

The Citation-Aware Research Assistant enables researchers to ask questions about scientific papers (e.g., "What are the key findings on GNN-Ret?") and receive concise, citation-supported answers. Built for Google Colab, it processes PDFs using advanced techniques like citation-aware chunking, section-aware splitting, semantic clustering, and HyDE-enhanced multi-hop retrieval, powered by the Groq API (llama-3.3-70b-versatile). This project aligns with insights from the GNN-Ret paper, offering a practical alternative to graph-based retrieval for scientific Q&A.

Why? To streamline literature review by preserving citation context and delivering accurate, evidence-based answers.

How? The pipeline ingests PDFs, splits them into meaningful chunks, computes embeddings, retrieves relevant sections, and generates answers using few-shot prompting.

# Features

Citation-Aware Chunking: Preserves claim-citation relationships using regex and position-aware splitting.

Section-Aware Splitting: Tags chunks with sections (e.g., Abstract, Methods) for context-aware retrieval.

Semantic Chunking: Clusters sentences by similarity using KMeans and sentence-transformers.

HyDE-Enhanced Retrieval: Generates hypothetical answers to improve query relevance.

Multi-Hop Retrieval: Iteratively refines queries for complex questions.

Interactive Interface: User-friendly Q&A interface in Colab using ipywidgets.

Groq API Integration: Leverages llama-3.3-70b-versatile for fast, high-quality responses.


⚙️ Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/username/research-assistant.git
cd research-assistant
2. Install Dependencies
bash
Copy
Edit
pip install -r src/requirements.txt
Required: PyPDF2, langchain, sentence-transformers, faiss-cpu, requests, ipywidgets, pdfplumber, scikit-learn

