
# YouTube RAG Chatbot

An RAG-Powered assistant that allows users to chat with any YouTube video. The application fetches the video transcript, processes it into a searchable knowledge base, and uses an LLM to answer questions grounded in the video's content.

## Screenshot


##  Features
- **Auto-Transcript Fetching:** Automatically retrieves captions using `youtube-transcript-api`.
- **RAG Pipeline:** Built with LangChain, utilizing `RecursiveCharacterTextSplitter` for document chunking.
- **Vector Search:** Uses FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`) for efficient semantic retrieval.
- **Interactive UI:** A clean, modular Streamlit interface for seamless chatting.

##  Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** OpenAI GPT-4o
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** HuggingFace Transformers

##  Project Structure
- `main.py`: The primary entry point for the application. This script contains the Streamlit user interface, manages the chat session state, and executes the RAG pipeline logic.

- `main.ipynb`: The research and development notebook. It serves as the technical documentation and experimental sandbox where the core indexing, retrieval, and chain-building logic was prototyped and tested.

- `.env`: (Local only) Stores sensitive credentials like the OPENAI_API_KEY to ensure security during development.

requirements.txt: Lists all Python dependencies required to run the application on both local and cloud environments.
- `requirements.txt`: List of necessary Python packages.

## Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API Key

### Installation
1. **Clone the repository:**
   ```bash
   git clone(https://github.com/your-username/youtube-chatbot.git)
   cd youtube-chatbot

2. **Set up a virtual environment:**
  ```bash
    python -m venv myenv
    On Windows: myenv\Scripts\activate

3.**Install dependencies**
   ```bash
    pip install -r requirements.txt

4.**Configure Environment Variables**:Create a .env file in the root directory and add your key:
     ```bash
    OPENAI_API_KEY=your_actual_key_here

5.**Running the App**
    using terminal:
     ```bash
    streamlit run frontend.py

### Future Improvements
Evaluation & Monitoring

Ragas Integration

LangSmith Tracing

Advanced Retrieval Strategies

Pre-Retrieval Optimization

Query Rewriting

Multi-Query Generation

During-Retrieval Refinement

Maximal Marginal Relevance (MMR)

Hybrid Search

Post-Retrieval Processing

Reranking

Contextual Compression

Augmentation & Generation

Answer Grounding

Context Window Optimization

Indexing Improvements

Semantic Chunking

Hierarchical Indexing

