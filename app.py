import streamlit as st
import os
import shutil
from dotenv import load_dotenv, set_key

# Import your custom modules
# Ensure the 'src' folder is in the same directory as this app.py or in the python path
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- Page Config ---
st.set_page_config(page_title="RAG Document Assistant", layout="wide")

# --- Session State Initialization ---
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

# --- Helper Functions ---
def save_api_key(key):
    """
    Saves the API key to .env file and sets it in the OS environment.
    """
    env_file_path = ".env"
    # Create .env if it doesn't exist
    if not os.path.exists(env_file_path):
        open(env_file_path, "w").close()
    
    # Write to .env
    set_key(env_file_path, "GROQ_API_KEY", key)
    # Set in current session
    os.environ["GROQ_API_KEY"] = key
    return True

def save_uploaded_file(uploaded_file, save_dir="data"):
    """
    Saves the uploaded streamlit file to the 'data' directory 
    so load_all_documents can read it.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Clear existing files in data folder if you want to start fresh per upload
    # or keep them to append. Here we clear to keep it simple.
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f'Failed to delete {file_path}. Reason: {e}')

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_dir

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key Input
    api_key_input = st.text_input("Enter Groq API Key", type="password", help="This will be stored in your .env file")
    
    if st.button("Save API Key"):
        if api_key_input:
            save_api_key(api_key_input)
            st.success("API Key saved to .env and environment set!")
        else:
            st.error("Please enter a valid key.")

    st.markdown("---")

    # 2. File Upload
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "md"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # A. Save file to disk so 'load_all_documents' can find it
                    data_dir = save_uploaded_file(uploaded_file)
                    
                    # B. Load documents using your backend
                    docs = load_all_documents(data_dir)
                    st.write(f"Loaded {len(docs)} document chunks.")

                    # C. Build Vector Store
                    store = FaissVectorStore("faiss_store")
                    store.build_from_documents(docs)
                    
                    st.session_state.vector_store_ready = True
                    st.success("Vector Store Index Built Successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# --- Main Content: RAG Search ---
st.title("🤖 RAG Summarizer")
st.markdown("Ask questions based on the documents you uploaded.")

# Check if API key is loaded
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ Please enter and save your Groq API Key in the sidebar to proceed.")
elif not st.session_state.vector_store_ready:
    st.info("👈 Please upload and process a document in the sidebar to start.")
else:
    # Query Interface
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching and generating summary..."):
                try:
                    # Initialize search with the store that was just built
                    # Note: RAGSearch internally should load the store or use the existing one
                    rag_search = RAGSearch() 
                    
                    # Perform search
                    summary = rag_search.search_and_summarize(query, top_k=3)
                    
                    st.subheader("Summary Result:")
                    st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")