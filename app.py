import os
import sys
import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict, Optional, Annotated, TypedDict

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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

# Define the state for our LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: Optional[str]
    retrieval_documents: Optional[List[Document]]
    results: Optional[str]

# Initialize the memory saver for persistence
memory = MemorySaver()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Reranking LLM with lower temperature
reranker_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# Define the nodes for our graph

def extract_query(state: State) -> Dict:
    """Extract the query from the latest message"""
    messages = state["messages"]
    if not messages:
        return {"query": None}
    
    # Get the most recent user message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or msg.get("role") == "user":
            query = msg.content if hasattr(msg, "content") else msg.get("content", "")
            debug(f"Extracted query: {query}")
            return {"query": query}
    
    return {"query": None}

def retrieve_documents(state: State) -> Dict:
    """Retrieve and rerank documents from the vector store"""
    query = state.get("query")
    if not query:
        debug("No query to retrieve documents for")
        return {"retrieval_documents": []}
    
    debug(f"Retrieving documents for query: {query}")
    
    # Load the vector store
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Get initial documents
    docs = vectorstore.similarity_search(query, k=10)
    debug(f"Retrieved {len(docs)} initial documents")
    
    if not docs:
        return {"retrieval_documents": []}
    
    if len(docs) <= 5:
        debug(f"Only {len(docs)} documents retrieved, skipping reranking")
        return {"retrieval_documents": docs}
    
    # Format documents for reranking
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append(f"Document {i+1}: {doc.page_content}")
    
    # Rerank with Gemini
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
        rerank_result = reranker_llm.invoke(rerank_prompt)
        debug(f"Reranking response: {rerank_result.content}")
        
        # Extract indices from the response
        result_content = rerank_result.content.strip()
        start_idx = result_content.find("[")
        end_idx = result_content.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            indices_str = result_content[start_idx:end_idx+1]
            try:
                import json
                ordered_indices = json.loads(indices_str)
                debug(f"Parsed reranking indices: {ordered_indices}")
                
                # Validate indices
                valid_indices = [idx-1 for idx in ordered_indices if 1 <= idx <= len(docs)]
                
                # Reorder documents
                reranked_docs = [docs[idx] for idx in valid_indices[:5]]
                debug(f"Returning {len(reranked_docs)} reranked documents")
                return {"retrieval_documents": reranked_docs}
            except Exception as e:
                debug(f"Error parsing reranking indices: {e}")
        
        debug("Reranking failed, returning top-k original documents")
        return {"retrieval_documents": docs[:5]}
        
    except Exception as e:
        debug(f"Error during reranking: {e}")
        return {"retrieval_documents": docs[:5]}

def generate_response(state: State) -> Dict:
    """Generate a response using the retrieved documents as context"""
    messages = state.get("messages", [])
    docs = state.get("retrieval_documents", [])
    query = state.get("query", "")
    
    if not query or not docs:
        debug("No query or documents to generate response from")
        # Just use the raw LLM if no context
        return {"messages": [llm.invoke(messages)]}
    
    # Create a context string from the documents
    context_str = "\n\n".join([doc.page_content for doc in docs])
    
    # Custom prompt template
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
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    formatted_prompt = prompt.format(context=context_str, question=query)
    debug("Sending prompt to Gemini")
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    content = response.content
    
    # Create AIMessage
    ai_message = AIMessage(content=content)
    
    return {"messages": [ai_message]}

# Build the graph
def create_graph():
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("extract_query", extract_query)
    graph_builder.add_node("retrieve_documents", retrieve_documents)
    graph_builder.add_node("generate_response", generate_response)
    
    # Add edges
    graph_builder.add_edge(START, "extract_query")
    graph_builder.add_edge("extract_query", "retrieve_documents")
    graph_builder.add_edge("retrieve_documents", "generate_response")
    graph_builder.add_edge("generate_response", END)
    
    # Compile the graph with memory
    return graph_builder.compile(checkpointer=memory)

# Initialize the graph
graph = create_graph()

# Function to process user input and generate responses
def process_message(message, history):
    debug(f"Received message: {message}")
    
    try:
        # Generate a unique thread ID based on the session
        thread_id = history[0][0] if history and history[0] else "new_session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Process the message through the graph
        debug("Processing message through graph")
        response = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config
        )
        debug("Response generated successfully")
        
        # Extract the AI response
        ai_response = response["messages"][-1].content
        return ai_response
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        debug(f"ERROR: {error_msg}")
        return error_msg

# Create the Gradio interface
def create_interface():
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
    return bot

# Launch the Gradio interface
if __name__ == "__main__":
    # Get ChromaDB path from environment variables
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")
    debug(f"Using ChromaDB from: {CHROMA_PATH}")
    debug("Launching Gradio interface")
    bot = create_interface()
    bot.launch(share=True)
