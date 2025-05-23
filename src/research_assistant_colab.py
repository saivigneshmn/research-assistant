import os
import re
import PyPDF2
import pdfplumber
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import faiss
import numpy as np
from typing import List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output
from functools import lru_cache

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
        self.hyde_cache = {}

    def extract_citations(self, text: str) -> List[Tuple[int, re.Match]]:
        """Extract citation markers and their positions."""
        citation_pattern = r'\[\d+(?:,\d+)*\]|\(\w+,\s*\d{4}\)'
        return [(m.start(), m) for m in re.finditer(citation_pattern, text)]

    def extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract sections using regex for common headers."""
        section_pattern = r'^(Abstract|Introduction|Methods|Results|Discussion|Conclusion)\s*$'
        sections = []
        current_section = None
        current_content = []

        for line in text.split('\n'):
            if re.match(section_pattern, line.strip(), re.IGNORECASE):
                if current_section and current_content:
                    sections.append((current_section, ' '.join(current_content)))
                current_section = line.strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section and current_content:
            sections.append((current_section, ' '.join(current_content)))

        return sections

    def semantic_chunking(self, text: str) -> List[str]:
        """Cluster sentences by semantic similarity."""
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if not sentences:
            return [text]
        
        sentence_embeddings = self.primary_embedder.encode(sentences, convert_to_numpy=True)
        num_clusters = max(1, len(sentences) // 5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sentence_embeddings)
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(sentences[i])
        
        return ['. '.join(cluster) + '.' for cluster in clusters if cluster]

    def citation_aware_chunking(self, text: str) -> List[str]:
        """Split text into citation-aware chunks with section and semantic awareness."""
        sections = self.extract_sections(text)
        final_chunks = []

        for section_name, section_text in sections:
            semantic_chunks = self.semantic_chunking(section_text)
            
            for chunk in semantic_chunks:
                citations = self.extract_citations(chunk)
                current_chunk = ""
                current_length = 0
                
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
                            final_chunks.append(f"{section_name}: {current_chunk}")
                            current_chunk = context_chunk
                            current_length = len(context_chunk)
                else:
                    if current_length + len(chunk) <= self.chunk_size:
                        current_chunk += chunk
                        current_length += len(chunk)
                    else:
                        final_chunks.append(f"{section_name}: {current_chunk}")
                        current_chunk = chunk
                        current_length = len(chunk)
                
                if current_chunk:
                    final_chunks.append(f"{section_name}: {current_chunk}")

        for i, chunk in enumerate(final_chunks):
            citations = self.extract_citations(chunk)
            self.citation_map[i] = [c[1].group() for c in citations]

        return final_chunks

    def ingest_pdfs(self) -> None:
        """Read PDFs, extract text, perform advanced chunking, and compute embeddings."""
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

    @lru_cache(maxsize=100)
    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a cached hypothetical answer using Groq API for HyDE."""
        prompt = f"Provide a brief hypothetical answer to the question: {query}"
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.groq_api_key}"
                },
                data=json.dumps({
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                })
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error generating hypothetical answer: {str(e)}")
            return query

    def extract_new_terms(self, chunks: List[Tuple[str, float, int]]) -> str:
        """Extract key terms from chunks for multi-hop refinement."""
        terms = []
        for chunk, _, _ in chunks:
            words = chunk.split()
            terms.extend([w for w in words if w.isalpha() and len(w) > 4])
        return ' '.join(list(set(terms))[:5])  # Convert set to list before slicing

    def multi_hop_retrieval(self, query: str, k: int = 3, depth: int = 2) -> List[Tuple[str, float, int]]:
        """Perform multi-hop retrieval by iteratively refining the query."""
        current_query = query
        all_chunks = []
        
        for _ in range(depth):
            chunks = self.similarity_search(current_query, k)
            all_chunks.extend(chunks)
            new_terms = self.extract_new_terms(chunks)
            current_query = f"{current_query} {new_terms}"
        
        unique_chunks = {chunk: (dist, idx) for chunk, dist, idx in all_chunks}
        sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1][0])[:k]
        return [(chunk, dist, idx) for chunk, (dist, idx) in sorted_chunks]

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float, int]]:
        """Perform similarity search with HyDE-style enhancement."""
        hypothetical_answer = self.generate_hypothetical_answer(query)
        combined_query = f"{query} {hypothetical_answer}"
        query_embedding = self.primary_embedder.encode([combined_query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        return [(self.chunks[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, float, int]]) -> str:
        """Generate a concise Q&A response using Groq API with citations."""
        few_shot_prompt = """
        **Example 1**
        Question: What is the main cause of climate change according to recent studies?
        Answer: Recent studies identify greenhouse gas emissions, particularly CO2 from fossil fuel combustion, as the primary cause of climate change [1,2].

        **Example 2**
        Question: How does the new algorithm improve neural network training?
        Answer: The algorithm enhances training by reducing runtime by 30% through optimized memory usage and improving accuracy with adaptive learning rates (Smith, 2023).

        **Current Question**
        Question: {query}
        Context: {context}
        Answer: Provide a concise, direct answer to the question in 1-2 sentences, including relevant citations from the context.
        """
        context = ""
        for i, (chunk, _, idx) in enumerate(retrieved_chunks):
            citations = self.citation_map.get(idx, [])
            context += f"{chunk} {' '.join(citations)}\n"

        prompt = few_shot_prompt.format(query=query, context=context)

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.groq_api_key}"
                },
                data=json.dumps({
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200
                })
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            return answer.strip()
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"

    def process_query(self, query: str, use_multi_hop: bool = False) -> str:
        """Process a user query and return a Q&A response."""
        if self.index is None or not self.chunks:
            raise ValueError("No PDFs ingested. Run ingest_pdfs() first.")
        retrieved_chunks = self.multi_hop_retrieval(query) if use_multi_hop else self.similarity_search(query)
        answer = self.generate_answer(query, retrieved_chunks)
        return answer

# Interactive interface for Q&A
def run_query(query):
    """Run a query and display the answer."""
    clear_output(wait=True)
    print(f"Query: {query}")
    answer = assistant.process_query(query, use_multi_hop=True)
    print(f"Answer: {answer}")

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
    
    # Set up interactive interface
    query_input = widgets.Text(description="Query:", layout={'width': '80%'})
    button = widgets.Button(description="Submit")
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            run_query(query_input.value)

    button.on_click(on_button_clicked)
    display(query_input, button, output)
    
    # Run example query
    run_query("What are the key findings on proposed GNN-Ret?")

