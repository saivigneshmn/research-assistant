import os
import re
import PyPDF2
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple

class CitationAwareResearchAssistant:
    def __init__(self, pdf_dir: str, groq_api_key: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with PDF directory, Groq API key, and chunking parameters."""
        self.pdf_dir = pdf_dir
        self.groq_api_key = groq_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.primary_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.citation_map = {}

    def extract_citations(self, text: str) -> List[Tuple[int, re.Match]]:
        """Extract citation markers and their positions."""
        citation_pattern = r'\[\d+(?:,\d+)*\]|\(\w+,\s*\d{4}\)'
        return [(m.start(), m) for m in re.finditer(citation_pattern, text)]

    def citation_aware_chunking(self, text: str) -> List[str]:
        """Split text into chunks, ensuring citations stay with their context."""
        initial_chunks = self.text_splitter.split_text(text)
        final_chunks = []
        current_chunk = ""
        current_length = 0

        for chunk in initial_chunks:
            citations = self.extract_citations(chunk)
            if citations:
                for pos, citation_match in citations:
                    sentence_end = chunk.rfind('.', 0, pos) + 1
                    if sentence_end == 0:
                        sentence_end = pos
                    context_chunk = chunk[:sentence_end] + citation_match.group()
                    if current_length + len(context_chunk) <= self.chunk_size:
                        current_chunk += context_chunk
                        current_length += len(context_chunk)
                    else:
                        final_chunks.append(current_chunk)
                        current_chunk = context_chunk
                        current_length = len(context_chunk)
            else:
                if current_length + len(chunk) <= self.chunk_size:
                    current_chunk += chunk
                    current_length += len(chunk)
                else:
                    final_chunks.append(current_chunk)
                    current_chunk = chunk
                    current_length = len(chunk)

        if current_chunk:
            final_chunks.append(current_chunk)

        for i, chunk in enumerate(final_chunks):
            citations = self.extract_citations(chunk)
            self.citation_map[i] = [c[1].group() for c in citations]

        return final_chunks

    def ingest_pdfs(self) -> None:
        """Read PDFs, extract text, perform citation-aware chunking, and compute embeddings."""
        all_text = ""
        for pdf_file in os.listdir(self.pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text = page.extract_text() or ""
                            all_text += text + "\n"
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")

        self.chunks = self.citation_aware_chunking(all_text)
        self.embeddings = self.primary_embedder.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float, int]]:
        """Perform similarity search to find top-k relevant chunks."""
        query_embedding = self.primary_embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        return [(self.chunks[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, float, int]]) -> str:
        """Generate a 5-point summary using Groq API with citations."""
        few_shot_prompt = """
        **Example 1**
        Question: What is the main finding on climate change impacts?
        Summary:
        - Rising temperatures increase hurricane intensity by 20% by 2050 [1].
        - Coastal flooding risks rise due to sea-level increases [2].
        - Drought frequency in arid regions doubles by 2100 [1,2].
        - Ecosystem disruptions affect 30% of species [3].
        - Adaptation measures reduce economic losses by 15% [2].

        **Example 2**
        Question: How does the proposed algorithm improve performance?
        Summary:
        - Reduces runtime by 30% via optimized memory usage (Smith, 2023).
        - Improves accuracy by 10% with adaptive learning (Jones, 2022).
        - Lowers energy consumption in training by 25% (Smith, 2023).
        - Scales better for large datasets (Lee, 2021).
        - Enhances model stability under noisy inputs (Jones, 2022).

        **Current Question**
        Question: {query}
        Context: {context}
        Summary: Provide exactly 5 concise bullet points summarizing the key findings, each including relevant citations from the context.
        """
        context = ""
        for i, (chunk, _, idx) in enumerate(retrieved_chunks):
            citations = self.citation_map.get(idx, [])
            context += f"{chunk} {' '.join(citations)}\n"

        prompt = few_shot_prompt.format(query=query, context=context)

        # Call Groq API
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.groq_api_key}"
                },
                data=json.dumps({
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"

    def process_query(self, query: str) -> str:
        """Process a user query and return a summarized answer."""
        if self.index is None or not self.chunks:
            raise ValueError("No PDFs ingested. Run ingest_pdfs() first.")
        retrieved_chunks = self.similarity_search(query)
        answer = self.generate_answer(query, retrieved_chunks)
        return answer

# Example usage in Colab
if __name__ == "__main__":
    # Create pdfs directory
    os.makedirs("./pdfs", exist_ok=True)
    
    # Uncomment to upload PDFs manually
    """
    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded.keys():
        os.rename(filename, os.path.join("./pdfs", filename))
    """
    
    # Use Google Drive instead (uncomment and adjust path if needed)
    """
    from google.colab import drive
    drive.mount('/content/drive')
    pdf_dir = "/content/drive/MyDrive/pdfs"
    """
    
    # Input Groq API key
    from getpass import getpass
    groq_api_key = getpass("Enter your Groq API key: ")
    
    pdf_dir = "./pdfs"
    assistant = CitationAwareResearchAssistant(pdf_dir=pdf_dir, groq_api_key=groq_api_key)
    
    # Ingest PDFs
    print("Ingesting PDFs...")
    assistant.ingest_pdfs()
    
    # Example query
    query = "What are the key findings on proposed GNN-Ret?"
    print(f"Query: {query}")
    answer = assistant.process_query(query)
    print(f"Answer:\n{answer}")

