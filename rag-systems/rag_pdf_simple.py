import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import PyPDF2
import uuid
import requests

st.set_page_config(page_title="Shreeya's RAG System", layout="wide")

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

llm_model = "ollama"
embedding_model = "chroma"


class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Read PDF and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        """Split text into chunks"""
        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size

            # If not at the start, include overlap
            if start > 0:
                start = start - self.chunk_overlap

            # Get chunk
            chunk = text[start:end]

            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {"source": pdf_file.name},
                }
            )

            start = end

        return chunks


class SimpleRAGSystem:
    """Simple RAG implementation"""

    def __init__(self, embedding_model="chroma", llm_model="ollama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chroma_db")

        # Setup embedding function based on model
        self.setup_embedding_function()

        # Setup LLM (only Ollama Llama3.2 available now)
        if llm_model == "ollama":
            self.llm_url = "http://localhost:11434/v1/chat/completions"
        else:
            raise ValueError("Currently, only Ollama Llama3.2 is supported for LLM.")

        # Get or create collection with proper handling
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Setup the appropriate embedding function"""
        try:
            if self.embedding_model == "chroma":
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            else:
                raise ValueError("Currently, only Chroma embeddings are supported.")

        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def setup_collection(self):
        """Setup collection with proper dimension handling"""
        collection_name = f"documents_chroma"

        try:
            # Try to get existing collection first
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )

            except:
                # If collection doesn't exist, create new one
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": "chroma"},
                )
                st.success("Created new collection for Chroma embeddings")

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def add_documents(self, chunks):
        """Add documents to ChromaDB"""
        try:
            # Ensure collection exists
            if not self.collection:
                self.collection = self.setup_collection()

            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            st.success("Documents added successfully")
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        """Query documents and return relevant chunks"""
        try:
            # Ensure collection exists
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(query_texts=[query], n_results=n_results)
            if results and "documents" in results and results["documents"]:
                st.success(f"Found {len(results['documents'])} relevant documents.")
            else:
                st.warning("No relevant documents found.")

            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context):
        """Generate response using LLMLlama3.2 via Ollama"""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """

            headers = {"Content-Type": "application/json"}
            data = {
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            }

            # Sending POST request to Ollama API
            response = requests.post(self.llm_url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error if the request failed

            # Assuming the response contains 'choices' and 'message' as in OpenAI's structure
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama API: {str(e)}")
            return None

    def get_embedding_info(self):
        """Get information about current embedding model(Chroma)"""

        model_info = {
            "name": "Chroma default",
            "dimensions": 384,
            "model": self.embedding_model,
        }

        return model_info


def main():
    # Custom CSS to style the application
    st.markdown("""
        <style>
            /* General styles */
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f7fb;
                color: #333;
            }
            /* Title styles */
            .title-center {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #4A90E2;
            }
            /* Card styles */
            .card {
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            /* Chat bubble styles */
            .chat-bubble {
                background-color: #FFFFFF;
                color: #333;
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                font-size: 16px;
            }
            /* User input styles */
            .user-input {
                background-color: #e1f5fe;
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            /* Sidebar styles */
            .sidebar .sidebar-content {
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            /* Button styles */
            .stButton > button {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stButton > button:hover {
                background-color: #357ab8;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title: Centered
    st.markdown('<h1 class="title-center">🤖 Shreeya\'s RAG System</h1>', unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("How can I help you today?")
    st.sidebar.markdown("### About Me:")
    st.sidebar.info("""
    Welcome to **Shreeya's RAG (Retrieval-Augmented Generation) System**! 
    Upload any PDF document and interact with it by asking questions based on its content. 
    Our AI-powered system will provide insightful answers by extracting relevant information directly from your document. 
    Simply upload a PDF, ask a question, and let the system assist you!
    """)

    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = "chroma"
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    embedding_model = "chroma"  # Embedding model fixed to "Chroma"
    llm_model = "ollama"  # LLM model fixed to "Ollama Llama3.2"

    # Check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()  # Clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None  # Reset RAG system
        st.warning("Embedding model changed. Please re-upload your documents.")

    # Initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # Layout: Create a container for PDF upload and query section
    st.markdown('<div style="height: 70px;"></div>', unsafe_allow_html=True)  # Added space

    cols = st.columns([0.3, 0.1, 0.6])  # Create a two-column layout

    # Column 1: PDF upload section (30% width)
    with cols[0]:
        st.subheader("📄 Upload PDF Document")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")

        if pdf_file and pdf_file.name not in st.session_state.processed_files:
            # Process PDF
            processor = SimplePDFProcessor()
            with st.spinner("Processing PDF..."):
                try:
                    # Extract text
                    text = processor.read_pdf(pdf_file)
                    # Create chunks
                    chunks = processor.create_chunks(text, pdf_file)
                    # Add to database
                    if st.session_state.rag_system.add_documents(chunks):
                        st.session_state.processed_files.add(pdf_file.name)
                        st.success(f"Successfully processed {pdf_file.name}")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

        if not pdf_file or pdf_file.name not in st.session_state.processed_files:
            st.info("👆 Please upload a PDF document to get started!")

    # Column 2: Query section (70% width)
    with cols[2]:
        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("📚 Chat with your PDFs")
            query = st.text_input("Ask a question about your documents:")

            if query:
                with st.spinner("Generating response..."):
                    # Get relevant chunks
                    results = st.session_state.rag_system.query_documents(query)
                    if results and results["documents"]:
                        # Generate response
                        response = st.session_state.rag_system.generate_response(
                            query, results["documents"][0]
                        )

                        if response:
                            # Display results in chat bubble style
                            st.markdown(f'<div class="chat-bubble">📝 Answer: {response}</div>', unsafe_allow_html=True)

                            with st.expander("View Source Passages"):
                                for idx, doc in enumerate(results["documents"][0], 1):
                                    st.markdown(f'<div class="chat-bubble">**Passage {idx}:** {doc}</div>', unsafe_allow_html=True)
        else:
            # Welcome message
            st.subheader("👋 Welcome to Shreeya's RAG System!")

            # About section
            st.markdown("""
                **About the System:**
                This is Shreeya's RAG (Retrieval-Augmented Generation) system. You can upload any PDF document, 
                and the system will allow you to ask questions related to the content of the uploaded document.
                It leverages Chroma for embeddings and Ollama for generating responses to your queries.
                Upload your document and ask your questions!
                """)

if __name__ == "__main__":
    main()