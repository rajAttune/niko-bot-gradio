# RAG Chatbot with LangGraph

A conversational RAG (Retrieval-Augmented Generation) chatbot built with LangGraph, Gemini, and ChromaDB.

## Features

- Contextual conversation with memory of past interactions
- Query enhancement based on conversation history
- Document retrieval with dynamic strategies based on query type
- Document reranking for improved relevance
- First-person responses in the style of the knowledge base author
- References to source documents

## Architecture

The application is structured in three main files:

- `app.py`: Application entry point, configuration, and Gradio UI
- `graph.py`: LangGraph state definition and node functions
- `rag_utils.py`: RAG utilities including document retrieval, reranking, and formatting

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the following variables:
   ```
   GOOGLE_API_KEY=your_google_api_key
   CHROMA_PATH=./chromadb
   DEBUG_LEVEL=1
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini
- `CHROMA_PATH`: Path to your ChromaDB vector database
- `DEBUG_LEVEL`: Debug level (0 = no debug, 1 = basic debug info)

## How It Works

1. The chatbot processes user queries through a LangGraph workflow
2. User queries are enhanced based on conversation history
3. The enhanced query is used to retrieve relevant documents from ChromaDB
4. Documents are reranked for relevance
5. The top documents are used as context for generating a response
6. The response is presented in the first person with references

## Query Types

The system recognizes three types of queries:
- **New topic**: A fresh question unrelated to previous conversation
- **Follow-up**: A question that extends a previous topic
- **Clarification**: A request for more detail about a previous response

Each query type triggers different retrieval and enhancement strategies.

## Extending

To extend the chatbot with new capabilities:
1. Add new utility functions to `rag_utils.py`
2. Create new node functions in `graph.py`
3. Update the graph structure in `app.py`

## License

[MIT License](LICENSE)
