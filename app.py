import streamlit as st
import os
import re
from dotenv import load_dotenv

# LangChain & API Imports
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Setup
load_dotenv()
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")

def extract_video_id(url):
    """Extracts the 11-character video ID from various YouTube URL formats."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    """Joins retrieved document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# 2. Sidebar: Video Processing Logic
with st.sidebar:
    st.title("🎥 Video Settings")
    youtube_url = st.text_input("Paste YouTube Link:")
    
    if st.button("Process Video"):
        video_id = extract_video_id(youtube_url)
        if video_id:
            with st.spinner("Analyzing transcript and building index..."):
                try:
                    # Logic matched to your working .ipynb file
                    # Initialize the API instance
                    ytt_api = YouTubeTranscriptApi() 
                    
                    # Use .fetch() as it is the method working for you
                    fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
                    
                    # Convert to raw data list of dicts and join text
                    transcript_list = fetched_transcript
                    transcript_text = " ".join(chunk.text for chunk in transcript_list)
                    
                    # Text Splitting
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.create_documents([transcript_text])
                    
                    # Embeddings & Vector Store (Local & Free)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    # Save retriever to session state so it persists across reruns
                    st.session_state.retriever = vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 4}
                    )
                    st.session_state.processed = True
                    st.success("Video processed! You can now ask questions.")
                    
                except TranscriptsDisabled:
                    st.error("Captions are disabled for this video.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a valid YouTube URL.")

# 3. Main Chat Interface
st.title("🤖 Chat with Video")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt_input := st.chat_input("Ask a question about the video"):
    if "retriever" not in st.session_state:
        st.warning("Please process a video link in the sidebar first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Build the Chain (Logic from your Notebook Cell In[55])
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        prompt_template = PromptTemplate(
            template="""
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.

              {context}
              Question: {question}
            """,
            input_variables=['context', 'question']
        )
        
        # LCEL Parallel Chain
        parallel_chain = RunnableParallel({
            "context": st.session_state.retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        full_chain = parallel_chain | prompt_template | llm | StrOutputParser()

        # Generate Assistant Response
        with st.chat_message("assistant"):
            response = full_chain.invoke(prompt_input)
            st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})