# backend.py
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_video_retriever(video_id):
    """Core logic extracted from your working notebook."""
    # Fetch
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=['en'])
    transcript = " ".join(chunk.text for chunk in fetched) # Using .text for objects
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    # Embed & Index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})