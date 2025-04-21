# Ollama RAG GUI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Ollama-powered-red.svg" alt="Powered by Ollama">
</p>

A powerful desktop application for building and querying your personal knowledge base using Ollama's language models. This application enables document retrieval with semantic search and AI-powered responses through a user-friendly graphical interface.

## 🌟 Features

- **Seamless Document Integration**: Import and process both Markdown (.md) and text (.txt) files
- **Intelligent Text Processing**: Automatically chunks documents with configurable overlap for optimal context retrieval
- **Vector Embeddings**: Creates semantic embeddings using Ollama's models to power accurate search
- **Intuitive GUI**: Clean desktop interface for all operations with no coding required
- **Multi-Model Support**: Automatically detects and uses all available models in your Ollama installation
- **Thread-Safe Architecture**: Background processing for resource-intensive tasks keeps the UI responsive
- **Chat-Style Interface**: Ask questions and receive AI-generated responses in a familiar chat format
- **Detailed Logging**: Comprehensive logging system with color-coded severity levels

## 🔍 How It Works

The application implements a complete Retrieval-Augmented Generation (RAG) workflow:

1. **Document Indexing**:
   - Select specific files or entire directories with Markdown/text content
   - Process text with intelligent chunking that preserves context between chunks
   - Generate semantic vector embeddings that capture the meaning of your documents
   - Save the knowledge base as a portable JSON file

2. **Knowledge Querying**:
   - Load your knowledge base index
   - Enter questions in natural language
   - Behind the scenes, the app:
     - Converts your question to a vector embedding
     - Finds the most relevant document chunks using cosine similarity
     - Constructs a prompt with the retrieved context
     - Generates a response using your selected Ollama model

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Ollama installed and running on your system ([Install Ollama](https://ollama.ai/))
- Required Python packages

### Step-by-Step Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ollama-rag-gui.git
   cd ollama-rag-gui
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

4. Pull recommended models (optional):
   ```bash
   ollama pull all-minilm  # Good for embeddings
   ollama pull llama3      # Good for responses
   ```

## 📖 Usage

### GUI Mode

Launch the graphical interface:

```bash
python ollama_rag_gui.py
```

**Indexing Documents:**
1. Go to the "Index Documents" tab
2. Select input directory or specific files
3. Choose an output file to save the index
4. Select an embedding model
5. Click "Index Documents"

**Querying Your Knowledge Base:**
1. Go to the "Query Knowledge Base" tab
2. Load your index file
3. Select a generation model
4. Type your question and click "Send"

### Command Line Mode

For advanced users or scripting:

**Index Documents:**
```bash
python ollama_rag.py index --input ./my_docs/ --output knowledge_base.json --embedding-model all-minilm
```

**Query the Knowledge Base:**
```bash
python ollama_rag.py query --index knowledge_base.json --query "What is RAG?" --model llama3
```

**Interactive Query Mode:**
```bash
python ollama_rag.py query --index knowledge_base.json --interactive
```

## 🧰 Technical Details

- Built with Python using Tkinter for cross-platform GUI compatibility
- Implements threading to keep the UI responsive during resource-intensive operations
- Uses NLTK for intelligent sentence tokenization
- Communicates with Ollama API for embeddings and text generation
- Stores document embeddings in a structured JSON format for portability

## 🎯 Use Cases

- Creating a searchable knowledge base from documentation
- Building a personal assistant for your notes and research
- Implementing a question-answering system for your data
- Learning how RAG systems work with a practical implementation

## 🔄 Advanced Customization

### Prompt Templates

You can customize the prompt template used for generation by modifying the `generate_response` method in `ollama_rag.py`:

```python
prompt_template = """
You are a helpful assistant with access to the following information:

{context}

Answer the following question based on the information above:
{query}

If the information to answer the question is not contained in the provided documents, say so clearly.
"""
```

### Chunking Parameters

Adjust chunking parameters for different document types by modifying the `_chunk_text` method in `ollama_rag.py`:

```python
def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100):
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing the local model API
- [NLTK](https://www.nltk.org/) for text processing capabilities

---

*Built with ❤️ for the open-source AI community* 
