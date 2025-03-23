import os
import re
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def extract_frontmatter(text):
    """
    Extract YAML frontmatter from Markdown text.
    
    Args:
        text: The Markdown text
        
    Returns:
        A tuple of (metadata_dict, content_without_frontmatter)
    """
    # Pattern to match YAML frontmatter
    pattern = r"^---\n(.*?)\n---\n(.*)$"
    match = re.match(pattern, text, re.DOTALL)
    
    if match:
        frontmatter_text = match.group(1)
        content = match.group(2)
        
        # Extract title and date
        metadata = {}
        title_match = re.search(r"title:\s*(.*?)$", frontmatter_text, re.MULTILINE)
        date_match = re.search(r"date:\s*(.*?)$", frontmatter_text, re.MULTILINE)
        
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        if date_match:
            metadata["date"] = date_match.group(1).strip()
            
        return metadata, content
    
    # If no frontmatter, return empty metadata and original content
    return {}, text

def load_documents_with_metadata(data_path):
    """
    Load Markdown documents from a directory and extract metadata.
    
    Args:
        data_path: Path to directory containing Markdown documents
    
    Returns:
        List of Document objects with metadata
    """
    print(f"Loading Markdown documents from {data_path}...")
    
    # First get the file paths using DirectoryLoader
    try:
        file_paths = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.md'):
                    file_paths.append(os.path.join(root, file))
                    
        print(f"Found {len(file_paths)} Markdown files")
    except Exception as e:
        print(f"Error finding Markdown files: {str(e)}")
        raise ValueError(f"No valid Markdown documents found in {data_path}")
    
    # Now process each file to extract metadata
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract metadata from frontmatter
            metadata, text_content = extract_frontmatter(content)
            
            # Add source as metadata
            metadata["source"] = file_path
            
            # Create Document object
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    print(f"Successfully processed {len(documents)} documents with metadata")
    
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
    # Load documents with metadata
    print(f"Loading documents from {data_path}...")
    documents = load_documents_with_metadata(data_path)
    print(f"Loaded {len(documents)} documents")
    
    # Split the documents into chunks while preserving metadata
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Print sample metadata to verify
    if chunks:
        print("Sample chunk metadata:", chunks[0].metadata)
    
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