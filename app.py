import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

def load_rag_system(chromadb_path):
    """
    Load an existing ChromaDB vector database and set up the RAG system.
    
    Args:
        chromadb_path: Path to the existing ChromaDB database
    
    Returns:
        A ConversationalRetrievalChain using the loaded vectorstore
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
    
    # Create a retriever from the vector store
    debug("Creating retriever with k=5")
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
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

    from langchain.prompts import PromptTemplate

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
        retriever=retriever,
        memory=memory,
        verbose=DEBUG_LEVEL > 0,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    debug("RAG system setup complete")
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
        "What is the most important color in business and why?"
    ]
)

# Launch the Gradio interface
if __name__ == "__main__":
    # Get ChromaDB path from environment variables
    CHROMA_PATH = os.getenv("CHROMA_PATH", CHROMA_PATH)
    debug(f"Using ChromaDB from: {CHROMA_PATH}")
    debug("Launching Gradio interface")
    bot.launch(share=True)
