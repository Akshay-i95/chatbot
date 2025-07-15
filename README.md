# 🤖 **Edify AI Chatbot**

Professional AI-powered document analysis with beautiful web interface.

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python start.py
```

Visit: **http://localhost:8501**

## ✨ **Features**

- 🔍 **Smart Document Search** - Vector-based similarity search
- 🤖 **AI-Powered Responses** - OpenRouter LLM integration  
- 📚 **Source Attribution** - Always shows source documents
- 💬 **Conversation Memory** - Contextual multi-turn conversations
- 📊 **Performance Analytics** - Real-time metrics and statistics
- 🎨 **Beautiful Interface** - Professional Streamlit web app

## 🏗️ **Architecture**

```
📱 Web App (app.py)
    ↓
🤖 AI Chatbot (chatbot.py)  
    ↓
🗄️ Vector Database (vector_db.py)
    ↓
📄 PDF Processor (pdf_processor.py)
```

## 🔧 **Configuration**

Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## 🎯 **Production Ready**

- Clean, simple codebase
- Error handling and logging
- Scalable architecture  
- Professional UI/UX
- Performance optimized

## 📚 **Documentation**

See `DEPLOYMENT.md` for detailed setup instructions.
