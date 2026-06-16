# Medical Chatbot 🏥

A RAG-based medical question answering chatbot built with FastAPI, Groq LLM, Pinecone vector store, and HuggingFace embeddings.

## Features

- **RAG Pipeline**: Retrieval Augmented Generation for accurate medical information
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Free LLM**: Uses Groq's Llama 3.3 70B model (free tier)
- **Vector Search**: Pinecone for semantic search of medical documents
- **HuggingFace Embeddings**: Free embedding model (all-MiniLM-L6-v2)
- **Beautiful Web Interface**: Interactive chat UI with real-time responses

## Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment (venv/virtualenv)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/komal-gosavi/Medical-Chatbot.git
   cd Medical-Chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your API keys:
   - `PINECONE_API_KEY`: Get from [Pinecone Console](https://console.pinecone.io)
   - `GROQ_API_KEY`: Get from [Groq Console](https://console.groq.com)

5. **Run the server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the application**
   - Web UI: http://127.0.0.1:8000
   - API Docs: http://127.0.0.1:8000/docs
   - Health Check: http://127.0.0.1:8000/health

## API Endpoints

### `GET /`
Renders the web interface for the chatbot.

### `POST /chat`
Send a medical question and get a response.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/chat -d "msg=What+are+symptoms+of+diabetes"
```

**Response:**
```json
{
  "response": "Diabetes symptoms include..."
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "rag_chain_loaded": false
}
```

## Project Structure

```
Medical-Chatbot/
├── app.py                 # FastAPI main application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── render.yaml           # Render deployment config
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration loader
│   ├── helper.py         # PDF loading and text splitting
│   ├── rag_chain.py      # RAG pipeline implementation
├── templates/
│   └── chat.html         # Web interface
└── data/                 # Medical documents (PDFs)
```

## Configuration

Edit `src/config.py` to customize:
- Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- Groq model (default: `llama-3.1-8b-instant`)
- Pinecone index name (default: `medical-chatbot`)
- Server settings (HOST, PORT, DEBUG)

## Technologies

- **Backend**: FastAPI, Uvicorn
- **LLM**: Groq Llama 3.3 70B (free)
- **Vector Store**: Pinecone
- **Embeddings**: HuggingFace sentence-transformers
- **Framework**: LangChain
- **Frontend**: HTML5, JavaScript

## Deployment

### Render.com
The project includes `render.yaml` for easy deployment to Render.com

### Docker
```bash
docker build -t medical-chatbot .
docker run -p 8000:8000 --env-file .env medical-chatbot
```

## Troubleshooting

### ImportError with LangChain
If you get `ModuleNotFoundError: No module named 'langchain.chains'`, ensure you have the correct versions:
```bash
pip install --upgrade langchain langchain-core langchain-community langchain-text-splitters
```

### Groq Model Not Found
Update to the latest Groq model in `src/config.py`:
```python
"GROQ_MODEL": "llama-3.1-8b-instant"
```

### Pinecone Connection Issues
- Verify your API key is correct in `.env`
- Ensure your Pinecone index exists and is configured
- Check Pinecone console for index status

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please create an issue on GitHub.

## Author

Created by Komal Gosavi

---

**Disclaimer**: This is an educational AI tool and should not replace professional medical advice. Always consult a qualified healthcare provider for medical concerns.
