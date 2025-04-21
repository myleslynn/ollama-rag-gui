# Ollama RAG for Document Files

This Python script implements a Retrieval-Augmented Generation (RAG) system for Ollama that works with markdown and text files. It allows you to:

1. Load and process markdown (.md) and text (.txt) files
2. Create embeddings using Ollama's embedding models
3. Query your document collection using natural language
4. Get responses augmented with relevant document context

## Requirements

- Python 3.7+
- Ollama installed and running on your system
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Make sure you have Ollama installed. If not, download it from [ollama.ai](https://ollama.ai).
2. Clone or download this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running:

```bash
ollama serve
```

5. Pull the required models (if you don't have them already):

```bash
ollama pull nomic-embed-text  # For embeddings
ollama pull llama3            # For generation (or any other model you prefer)
```

## Usage

The script provides two main commands: `index` and `query`.

### Indexing Document Files

To index your document files:

```bash
python ollama_rag.py index --input ./docs/ --output knowledge_base.json
```

Options:
- `--input`, `-i`: Input directory or file pattern (e.g., "./docs/*.md" or "./docs/*.txt")
- `--output`, `-o`: Output JSON file to save the indexed documents
- `--chunk-size`: Maximum chunk size in characters (default: 1000)
- `--embedding-model`: Embedding model to use (default: "nomic-embed-text")

### Querying the Knowledge Base

To query your knowledge base:

```bash
python ollama_rag.py query --index knowledge_base.json --query "What is RAG?"
```

For interactive mode:

```bash
python ollama_rag.py query --index knowledge_base.json --interactive
```

Options:
- `--index`, `-i`: Path to the JSON file containing your indexed documents
- `--query`, `-q`: Question to ask
- `--model`, `-m`: LLM model to use (default: "llama3")
- `--interactive`: Enable interactive mode for multiple queries

## Graphical User Interface

The system also includes a GUI for easier operation:

```bash
python ollama_rag_gui.py
```

The GUI allows you to:
- Select directories or specific files for indexing
- Choose from available Ollama models
- Set chunking parameters
- Query your knowledge base with a chat-like interface

## How It Works

1. **Document Processing**:
   - Loads markdown and text files from the specified location
   - Converts markdown to plain text (text files are used directly)
   - Splits text into overlapping chunks of specified size

2. **Embedding Creation**:
   - Creates vector embeddings for each document chunk using Ollama's embedding model
   - Stores the embeddings along with document metadata

3. **Retrieval**:
   - When you ask a question, it creates an embedding for your query
   - Finds the most similar document chunks using cosine similarity
   - Retrieves the top-k most relevant chunks

4. **Generation**:
   - Constructs a prompt containing the retrieved document chunks and your question
   - Sends the prompt to Ollama to generate a contextually informed response

## Customization

You can customize the behavior by:
- Adjusting chunk size for different document lengths
- Using different embedding or LLM models
- Modifying the prompt template in the code

## Example

```bash
# Index a directory of mixed markdown and text files
python ollama_rag.py index --input ./documentation/ --output docs.json

# Query in interactive mode
python ollama_rag.py query --index docs.json --interactive --model mistral
```

## License

MIT 