import os
import sys
import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set debug level (0 = no debug, 1 = basic debug info)
DEBUG_LEVEL = 1

def debug(message):
    """Print debug message to stderr if debug level is enabled"""
    if DEBUG_LEVEL > 0:
        print(f"DEBUG: {message}", file=sys.stderr)

from langchain_core.pydantic_v1 import Field

class GeminiRerankerRetriever(BaseRetriever):
    """Custom retriever that first fetches documents and then reranks them using Gemini."""
    
    base_retriever: BaseRetriever = Field(..., description="Base retriever to get initial documents")
    reranker_llm: ChatGoogleGenerativeAI = Field(..., description="LLM used for reranking documents")
    k: int = Field(5, description="Number of documents to return after reranking")
    
    class Config:
        arbitrary_types_allowed = True
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents and rerank them with Gemini."""
        # Step 1: Get more documents than we need (we'll rerank and filter)
        initial_k = min(self.k * 2, 10)  # Get more docs, but set a reasonable limit
        docs = self.base_retriever.get_relevant_documents(query)
        debug(f"Retrieved {len(docs)} initial documents")
        
        if not docs:
            debug("No documents retrieved from base retriever")
            return []
        
        # If only a few docs, skip reranking
        if len(docs) <= self.k:
            debug(f"Only {len(docs)} documents retrieved, skipping reranking")
            return docs
            
        # Step 2: Format documents for reranking
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"Document {i+1}: {doc.page_content}")
        
        # Step 3: Rerank with Gemini
        rerank_prompt = f"""
        You are a document reranking system. Your task is to rerank the following documents based on their relevance to the query.
        
        Query: {query}
        
        Documents:
        {formatted_docs}
        
        Return a JSON list of document indices in descending order of relevance to the query.
        For example: [3, 1, 5, 2, 4] means document 3 is most relevant, followed by 1, etc.
        Only return the JSON list, no other text.
        """
        
        debug("Sending reranking request to Gemini")
        try:
            rerank_result = self.reranker_llm.invoke(rerank_prompt)
            debug(f"Reranking response: {rerank_result.content}")
            
            # Extract indices from the response
            # First, clean up the response - remove any text outside the brackets
            result_content = rerank_result.content.strip()
            start_idx = result_content.find("[")
            end_idx = result_content.rfind("]")
            if start_idx != -1 and end_idx != -1:
                indices_str = result_content[start_idx:end_idx+1]
                # Parse the indices - simple approach, assuming clean output
                try:
                    import json
                    ordered_indices = json.loads(indices_str)
                    debug(f"Parsed reranking indices: {ordered_indices}")
                    
                    # Validate indices
                    valid_indices = [idx-1 for idx in ordered_indices if 1 <= idx <= len(docs)]
                    
                    # Reorder documents
                    reranked_docs = [docs[idx] for idx in valid_indices[:self.k]]
                    debug(f"Returning {len(reranked_docs)} reranked documents")
                    return reranked_docs
                except Exception as e:
                    debug(f"Error parsing reranking indices: {e}")
                    # Fall back to original order if parsing fails
            
            debug("Reranking failed, returning top-k original documents")
            return docs[:self.k]
            
        except Exception as e:
            debug(f"Error during reranking: {e}")
            return docs[:self.k]

def load_rag_system(chromadb_path):
    """
    Load an existing ChromaDB vector database and set up the RAG system with reranking.
    
    Args:
        chromadb_path: Path to the existing ChromaDB database
    
    Returns:
        A ConversationalRetrievalChain using the loaded vectorstore with reranking
    """
    debug(f"Loading RAG system from ChromaDB at: {chromadb_path}")
    
    # Initialize the embeddings
    debug("Initializing text-embedding-004 embeddings")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load the existing Chroma database
    debug("Loading existing Chroma database")
    vectorstore = Chroma(
        persist_directory=chromadb_path,
        embedding_function=embeddings
    )
    debug(f"Vectorstore loaded with {vectorstore._collection.count()} vectors")
    
    # Create a base retriever from the vector store
    debug("Creating base retriever")
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}  # Retrieve more documents initially
    )
    
    # Initialize the reranker model
    debug("Initializing Gemini model for reranking")
    reranker_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,  # Use low temperature for more consistent ranking
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the reranker retriever
    debug("Creating reranker retriever")
    reranker_retriever = GeminiRerankerRetriever(
        base_retriever=base_retriever,
        reranker_llm=reranker_llm,
        k=5  # Final number of documents to return
    )
    
    # Set up memory for conversation history
    debug("Setting up conversation buffer memory")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Initialize the Gemini model for chat
    debug("Initializing Gemini model for chat")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    # Create a custom prompt that includes both the system instruction and retrieved documents
    template = """You are an assistant designed to represent Niko Canner, an 
    investor, entrepreneur, philosopher, and founder of Incandescent, a consulting firm in NYC.
    You respond based ONLY on what you find in your knowledge base, which are 
    blog posts written by Niko. If you don't find something in the knowledge base,
    just say so, and don't make up anything else.
    Summarize the documents you find and respond in first person, 
    balancing a conversational and professional tone. Your responses are a
    good balance of length and depth.

    RESPOND IN FIRST PERSON ONLY!!!

    Context information is below.
    ---------------------
    {context}
    ---------------------

    Given the context information and not prior knowledge, answer the question: {question}
    """

    custom_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Use the prompt in the chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=reranker_retriever,
        memory=memory,
        verbose=DEBUG_LEVEL > 0,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    debug("RAG system setup complete with reranking")
    return conversation_chain

# Initialize the conversation chain with the specified ChromaDB path
CHROMA_PATH = "./chromadb"
conversation_chain = None

# Function to process user input and generate responses
def process_message(message, history):
    global conversation_chain
    
    debug(f"Received message: {message}")
    
    # Initialize the conversation chain if not already done
    if conversation_chain is None:
        debug("Conversation chain not initialized, loading now...")
        try:
            conversation_chain = load_rag_system(CHROMA_PATH)
            debug("Conversation chain initialized successfully")
        except Exception as e:
            error_msg = f"Error loading the knowledge base: {str(e)}"
            debug(f"ERROR: {error_msg}")
            return error_msg
    
    try:
        # Process the message through the RAG system
        debug("Processing message through RAG system")
        response = conversation_chain.invoke({"question": message})
        debug("Response generated successfully")
        return response["answer"]
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        debug(f"ERROR: {error_msg}")
        return error_msg

# Create the Gradio interface
debug("Setting up Gradio ChatInterface")
bot = gr.ChatInterface(
    fn=process_message,
    title="Chat with Niko",
    description="Ask questions and get answers based on Niko's blog.",
    examples=[
        "Give me some tips to manage conflicts",
        "What is the most important color in business and why?",
        "How do I handle difficult conversations?"
    ]
)

# Launch the Gradio interface
if __name__ == "__main__":
    # Get ChromaDB path from environment variables
    CHROMA_PATH = os.getenv("CHROMA_PATH", CHROMA_PATH)
    debug(f"Using ChromaDB from: {CHROMA_PATH}")
    debug("Launching Gradio interface")
    bot.launch(share=True)