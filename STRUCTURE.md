# 📁 **PROJECT STRUCTURE**

## **Production-Ready File Organization**

```
edify-chatbot/
├── 📱 app.py              # Main Streamlit web application
├── 🤖 chatbot.py          # AI chatbot interface
├── 🗄️ vector_db.py        # Vector database manager
├── 📄 pdf_processor.py    # PDF processing engine
├── ⚙️ pipeline.py         # Data processing pipeline
├── 🔗 llm_service.py      # LLM integration service
├── 🚀 start.py            # Quick start script
├── 🔧 .env                # Configuration file
├── 📋 requirements.txt    # Dependencies
├── 📖 README.md           # Project overview
├── 🚀 DEPLOYMENT.md       # Deployment guide
├── 📂 vector_store/       # Vector database storage
├── 📝 *.log               # Application logs
└── 🗂️ __pycache__/        # Python cache
```

## **Core Components**

| File | Purpose | Dependencies |
|------|---------|-------------|
| `app.py` | Web interface | streamlit, chatbot.py |
| `chatbot.py` | AI engine | vector_db.py, llm_service.py |
| `vector_db.py` | Data storage | chromadb, sentence-transformers |
| `pdf_processor.py` | Document processing | PyMuPDF, pytesseract |
| `pipeline.py` | Data pipeline | pdf_processor.py, vector_db.py |
| `llm_service.py` | AI responses | openai (OpenRouter) |

## **Simple & Clean Design**

✅ **Removed "enhanced" prefixes**  
✅ **Consolidated duplicate files**  
✅ **Updated all import statements**  
✅ **Clean, descriptive names**  
✅ **Production-ready structure**
