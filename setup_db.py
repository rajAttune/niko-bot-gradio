import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def load_documents(data_path):
    """
    Load Markdown documents from a directory.
    
    Args:
        data_path: Path to directory containing Markdown documents
    
    Returns:
        List of loaded documents
    """
    print(f"Loading Markdown documents from {data_path}...")
    
    try:
        # Use the DirectoryLoader to load all Markdown files
        markdown_loader = DirectoryLoader(
            data_path, 
            glob="**/*.md", 
            loader_cls=UnstructuredMarkdownLoader
        )
        documents = markdown_loader.load()
        print(f"Loaded {len(documents)} Markdown files")
    except Exception as e:
        print(f"Error loading Markdown files: {str(e)}")
        raise ValueError(f"No valid Markdown documents found in {data_path}")
    
    if not documents:
        raise ValueError(f"No documents found in {data_path}")
    
    return documents

def create_rag_database(data_path, output_path, chunk_size=1000, chunk_overlap=200):
    """
    Create a RAG database from documents using Gemini embeddings.
    
    Args:
        data_path: Path to directory containing documents
        output_path: Path to save the ChromaDB database
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    # Load documents
    print(f"Loading documents from {data_path}...")
    documents = load_documents(data_path)
    print(f"Loaded {len(documents)} documents")
    
    # Split the documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings with Gemini
    print("Creating embeddings with Gemini...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    
    # Create and persist the vector database
    print(f"Creating vector database at {output_path}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=output_path
    )
    vectorstore.persist()
    
    print(f"Vector database created successfully at {output_path}")
    print(f"Total vectors: {vectorstore._collection.count()}")
    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a RAG database with Gemini embeddings")
    parser.add_argument("--data_path", required=True, help="Path to the directory containing documents")
    parser.add_argument("--output_path", required=True, help="Path to save the ChromaDB database")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    create_rag_database(
        args.data_path,
        args.output_path,
        args.chunk_size,
        args.chunk_overlap
    )