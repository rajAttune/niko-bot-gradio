import os
import sys
import gradio as gr
import re
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
    original_query: Optional[str]  # Store original query before enhancement
    retrieval_documents: Optional[List[Document]]
    results: Optional[str]
    history: List[Dict[str, str]]  # New field to store conversation history
    query_type: Optional[str]  # Store query classification (new, followup, clarification)

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
    """Extract the query from the latest message and enhance with history if needed"""
    messages = state["messages"]
    history = state.get("history", [])
    
    if not messages:
        return {"query": None}
    
    # Get the most recent user message
    current_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or msg.get("role") == "user":
            current_query = msg.content if hasattr(msg, "content") else msg.get("content", "")
            debug(f"Extracted current query: {current_query}")
            break
    
    if not current_query:
        return {"query": None}
        
    # Handle different query enhancement strategies based on query type
    enhanced_query = enhance_query_with_history(current_query, history)
    debug(f"Enhanced query: {enhanced_query}")
    
    return {
        "query": enhanced_query,
        "original_query": current_query  # Store original for prompt
    }

def retrieve_documents(state: State) -> Dict:
    """Retrieve and rerank documents from the vector store using enhanced query"""
    query = state.get("query")
    original_query = state.get("original_query", query)
    
    if not query:
        debug("No query to retrieve documents for")
        return {"retrieval_documents": []}
        
    # Classify query type
    history = state.get("history", [])
    last_exchange = history[-1] if history else None
    query_type = "new_topic"
    
    if last_exchange:
        query_type = classify_query_type(original_query, last_exchange)
        
    debug(f"Query type classified as: {query_type}")
    
    # Store query type in state
    state_updates = {"query_type": query_type}
    
    debug(f"Retrieving documents for query: {query}")
    debug(f"Query type: {query_type}")
    
    # Load the vector store
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Get initial documents based on query type
    docs = []
    
    if query_type == "new_topic":
        # For new topics, just use the current query
        docs = vectorstore.similarity_search(query, k=10)
        debug("Using standard retrieval for new topic")
        
    elif query_type == "followup":
        # For follow-ups, try hybrid approach
        # 1. Retrieve based on enhanced query
        enhanced_docs = vectorstore.similarity_search(query, k=7)
        
        # 2. Also retrieve based on original query with higher k value
        original_docs = vectorstore.similarity_search(original_query, k=5)
        
        # Combine documents with deduplication (prioritize enhanced results)
        seen_ids = set()
        docs = []
        
        # First add enhanced results
        for doc in enhanced_docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in seen_ids:
                docs.append(doc)
                seen_ids.add(doc_id)
        
        # Then add original results if not already included
        for doc in original_docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in seen_ids and len(docs) < 10:
                docs.append(doc)
                seen_ids.add(doc_id)
                
        debug(f"Using hybrid retrieval for follow-up question, found {len(docs)} documents")
        
    else:  # clarification
        # For clarifications, prioritize retrieval based on enhanced query
        docs = vectorstore.similarity_search(query, k=10)
        debug("Using enhanced retrieval for clarification")
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
    You are a document reranking system. 
    Your task is to rerank the following documents based on their relevance to the query.
    
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
                return {**state_updates, "retrieval_documents": reranked_docs}
            except Exception as e:
                debug(f"Error parsing reranking indices: {e}")
        
        debug("Reranking failed, returning top-k original documents")
        return {**state_updates, "retrieval_documents": docs[:5]}
        
    except Exception as e:
        debug(f"Error during reranking: {e}")
        return {"retrieval_documents": docs[:5]}

def format_sources(docs: List[Document]) -> str:
    """Format source documents in the required format: '* Title (date)'"""
    sources = []
    seen_titles = set()  # To track unique sources
    
    for doc in docs:
        # Extract metadata
        metadata = doc.metadata
        title = metadata.get("title", "Untitled Document")
        date = metadata.get("date", "No date")
        
        # Create source entry
        source_entry = f"* {title} ({date})"
        
        # Only add if we haven't seen this title before
        if title not in seen_titles:
            sources.append(source_entry)
            seen_titles.add(title)
    
    # Format as a string
    if sources:
        return "\n\nReferences:\n" + "\n".join(sources)
    else:
        return "\n\nReferences:\n* No specific sources found"

def format_chat_history(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """Format the chat history for inclusion in the prompt"""
    if not history:
        return ""
    
    # Take only the most recent exchanges, limited by max_turns
    recent_history = history[-max_turns:]
    
    formatted_history = []
    for exchange in recent_history:
        user_message = exchange.get("user", "")
        ai_message = exchange.get("assistant", "")
        
        # Remove references section from AI messages for cleaner history
        ai_message = re.sub(r'\n+References:.*?$', '', ai_message, flags=re.DOTALL)
        
        formatted_history.append(f"User: {user_message}")
        formatted_history.append(f"Assistant: {ai_message}")
    
    return "\n".join(formatted_history)

def enhance_query_with_history(query: str, history: List[Dict[str, str]]) -> str:
    """Enhance query with history based on query type detection
    
    Implements three strategies:
    1. For new topics: Use query as-is
    2. For follow-ups: Augment query with key terms from previous exchange
    3. For clarifications: Use recent history + query
    """
    if not history:
        return query
    
    # Get the last exchange
    last_exchange = history[-1] if history else None
    if not last_exchange:
        return query
    
    # Attempt to classify query type
    query_type = classify_query_type(query, last_exchange)
    
    # Apply enhancement strategy based on query type
    if query_type == "new_topic":
        # Independent question - use as is
        return query
        
    elif query_type == "followup":
        # Follow-up question - add key terms from last exchange
        last_user_query = last_exchange.get("user", "")
        key_terms = extract_key_terms(last_user_query)
        # Combine but prioritize current query
        return f"{query} (context: {key_terms})"
        
    elif query_type == "clarification":
        # Multi-turn clarification - use more history
        # Include user's previous query and key points from assistant's response
        last_user_query = last_exchange.get("user", "")
        last_response = last_exchange.get("assistant", "")
        # Get first sentence of response to capture main point
        first_sentence = last_response.split(".")[0] if last_response else ""
        return f"{query} (referring to: {last_user_query} - {first_sentence})"
        
    # Default fallback
    return query

def classify_query_type(query: str, last_exchange: Dict[str, str]) -> str:
    """Classify query as new topic, follow-up, or clarification
    
    Uses simple heuristics - could be replaced with more sophisticated methods
    """
    query = query.lower()
    last_query = last_exchange.get("user", "").lower()
    
    # Check for explicit reference words
    reference_terms = ["it", "that", "this", "those", "they", "he", "she", 
                       "the same", "the above", "mentioned", "previous"]
    
    # Check for follow-up indicators
    followup_indicators = ["more", "explain", "elaborate", "tell me more", 
                           "what about", "how about", "why", "expand"]
    
    # Simple reference detection
    for term in reference_terms:
        if term in query:
            return "clarification"
    
    # Follow-up detection
    for indicator in followup_indicators:
        if indicator in query:
            return "followup"
    
    # Check for overlapping terms (excluding common words)
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "with", 
                   "is", "are", "was", "were", "be", "to", "for", "of"}
    
    # Tokenize and filter
    query_terms = {term.lower() for term in query.split() 
                  if term.lower() not in common_words}
    last_terms = {term.lower() for term in last_query.split() 
                 if term.lower() not in common_words}
    
    # Calculate overlap
    overlap = query_terms.intersection(last_terms)
    overlap_ratio = len(overlap) / len(query_terms) if query_terms else 0
    
    # High overlap suggests clarification or follow-up
    if overlap_ratio > 0.3:
        return "followup"
    
    # Default to new topic
    return "new_topic"

def extract_key_terms(text: str) -> str:
    """Extract key terms from text, skipping common words"""
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "with", 
                   "is", "are", "was", "were", "be", "to", "for", "of", "you", 
                   "me", "what", "who", "how", "why", "when", "where"}
    
    # Simple extraction - keep words with 4+ chars that aren't common
    terms = [word for word in text.lower().split() 
             if len(word) >= 4 and word not in common_words]
    
    # Return top 3-5 terms
    return " ".join(terms[:5])

def generate_response(state: State) -> Dict:
    """Generate a response using the retrieved documents as context and chat history"""
    messages = state.get("messages", [])
    docs = state.get("retrieval_documents", [])
    query = state.get("original_query", state.get("query", ""))  # Use original query for response
    history = state.get("history", [])
    
    if not query or not docs:
        debug("No query or documents to generate response from")
        # Just use the raw LLM if no context
        ai_response = llm.invoke(messages)
        
        # Update history
        new_exchange = {"user": query, "assistant": ai_response.content}
        updated_history = history + [new_exchange]
        
        return {
            "messages": [ai_response],
            "history": updated_history
        }
    
    # Create a context string from the documents
    context_str = "\n\n".join([doc.page_content for doc in docs])
    
    # Format sources in the required format
    sources_str = format_sources(docs)
    
    # Format chat history (last 3 exchanges)
    history_str = format_chat_history(history, max_turns=3)
    
    # Get query type for prompt customization
    query_type = state.get("query_type", "new_topic")
    
    # Custom prompt template - modified to include chat history with query type awareness
    template = """You are an assistant designed to represent Niko Canner, an 
    investor, entrepreneur, philosopher, and founder of Incandescent, a consulting firm in NYC.
    You respond based ONLY on what you find in your knowledge base, which are 
    blog posts written by Niko. If you don't find something in the knowledge base,
    just say so, and don't make up anything else.
    
    Summarize the documents you find and respond in first person, 
    balancing a conversational and professional tone. Your responses are a
    good balance of length and depth.

    RESPOND IN FIRST PERSON ONLY!!!
    
    {history_section}
    
    Context information is below.
    ---------------------
    {context}
    ---------------------

    Given the context information and not prior knowledge, answer the question: {question}
    
    Query type: {query_type}
    """
    
    # Add history section if history exists
    history_section = ""
    if history_str:
        history_section = f"""Previous conversation:
    ---------------------
    {history_str}
    ---------------------"""
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question", "history_section", "query_type"]
    )
    
    formatted_prompt = prompt.format(
        context=context_str, 
        question=query, 
        history_section=history_section,
        query_type=query_type
    )
    debug("Sending prompt to Gemini")
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    content = response.content
    
    # Remove any references section that might have been generated despite instructions
    content = re.sub(r'\n+References:.*?$', '', content, flags=re.DOTALL)
    
    # Append our properly formatted sources
    final_content = content + sources_str
    
    # Create AIMessage
    ai_message = AIMessage(content=final_content)
    
    # Update history with the new exchange
    new_exchange = {"user": query, "assistant": final_content}
    updated_history = history + [new_exchange]
    
    return {
        "messages": [ai_message],
        "history": updated_history
    }

def initialize_state(state: State) -> Dict:
    """Initialize state with empty history if not present"""
    if "history" not in state:
        return {"history": []}
    return {}

# Build the graph
def create_graph():
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("initialize_state", initialize_state)
    graph_builder.add_node("extract_query", extract_query)
    graph_builder.add_node("retrieve_documents", retrieve_documents)
    graph_builder.add_node("generate_response", generate_response)
    
    # Add edges
    graph_builder.add_edge(START, "initialize_state")
    graph_builder.add_edge("initialize_state", "extract_query")
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
            "What is the most important color and why?",
            "How do I handle difficult conversations?",
            "What are the 6 Cs?"
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

