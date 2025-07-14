# Azure Blob Storage PDF Vectorization Pipeline

This project provides a comprehensive solution for processing PDF documents from Azure Blob Storage, extracting text, and creating a searchable vector database for semantic search and retrieval.

## 🚀 Features

- **Intelligent PDF Processing**: Stream PDFs from Azure Blob Storage without downloading
- **Concurrent Processing**: Multi-threaded processing with intelligent batching
- **Memory Efficient**: Configurable memory limits and chunk-based processing
- **Vector Database**: Support for ChromaDB and FAISS vector databases
- **Semantic Search**: Natural language search through your PDF collection
- **Rich Metadata**: Comprehensive metadata extraction and storage
- **Interactive Interface**: User-friendly search interface

## 📁 Project Structure

```
connection-test/
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
├── validate_environment.py   # Pre-flight validation script
├── azure_blob_test.py       # Azure Blob Storage connection test
├── pdf_processor.py         # PDF streaming and text extraction
├── vector_db_manager.py     # Vector database management
├── main_pipeline.py         # Main processing pipeline
├── interactive_search.py    # Interactive search interface
└── README.md               # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   
   Edit the `.env` file and add your Azure credentials. The script will try authentication methods in this priority order:

   ### Method 1: Connection String (PRIMARY - Recommended)
   This contains everything needed (account name, key, endpoint):
   ```
   AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=edifystorageaccount;AccountKey=YOUR_ACCOUNT_KEY;EndpointSuffix=core.windows.net
   ```

   ### Method 2: SAS Token (FALLBACK 1)
   If connection string is not available:
   ```
   AZURE_STORAGE_SAS_TOKEN=YOUR_SAS_TOKEN
   ```

   ### Method 3: Account Key (FALLBACK 2)
   If neither connection string nor SAS token work:
   ```
   AZURE_STORAGE_ACCOUNT_KEY=YOUR_ACCOUNT_KEY
   ```

   **Note:** The script will automatically try each method in order until one works.

3. **Set container and folder path:**
   ```
   AZURE_STORAGE_CONTAINER_NAME=edifydocumentcontainer-dev
   AZURE_BLOB_FOLDER_PATH=edipedia/2025-2026/
   ```

4. **Configure filtering and summary options:**
   ```
   FILTER_PDF_FILES=true        # Show only PDF files
   SHOW_FILE_SUMMARY=true       # Display detailed summary statistics
   ```

## Usage

Run the test script:
```bash
python azure_blob_test.py
```

## What it does

- ✅ Tests connection to Azure Blob Storage
- 📂 Lists all folders in the specified path
- 📄 Lists all files with their sizes and modification dates
- � Can filter to show only PDF files
- 📊 Provides detailed summary statistics including:
  - Total number of files and folders
  - PDF file count and total size
  - Percentage of PDFs vs all files
  - Storage usage breakdown
- �🔍 Provides clear error messages for troubleshooting

## 🎯 Quick Start

### 0. Validate Environment (Recommended First Step)
```bash
python validate_environment.py
```

### 1. Test Azure Connection
```bash
python azure_blob_test.py
```

### 2. Process PDFs and Build Vector Database
```bash
python main_pipeline.py
```

### 3. Search Your Documents
```bash
python interactive_search.py
```

## 🔧 Advanced Configuration

### Processing Settings
- `MAX_WORKERS`: Number of concurrent PDF processors (default: 8)
- `CHUNK_SIZE`: Text chunk size in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `BATCH_SIZE`: Files per processing batch (default: 50)
- `MAX_MEMORY_MB`: Memory limit in MB (default: 2048)

### Vector Database Settings
- `VECTOR_DB_TYPE`: Database type (chromadb/faiss, default: chromadb)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `COLLECTION_NAME`: Database collection name (default: pdf_documents)

## 📊 Performance

With your current setup (251 PDFs, 1175.35 MB):
- **Estimated Processing Time**: 15-30 minutes (depending on hardware)
- **Memory Usage**: 2-4 GB peak
- **Vector Database Size**: ~500 MB
- **Search Speed**: Sub-second queries

## 🔍 Search Examples

Once processed, you can search using natural language:
- "education policy and curriculum"
- "student assessment methods"
- "teaching strategies for mathematics"
- "classroom management techniques"

## 🛠️ Troubleshooting

### Common Issues:

1. **Authentication Error**: Check that your credentials are correct
2. **Container Not Found**: Verify the container name exists
3. **Access Denied**: Ensure your credentials have read permissions
4. **Network Issues**: Check your internet connection and firewall settings

### Getting Your Credentials:

1. **Connection String**: Azure Portal → Storage Account → Access Keys → Connection String
2. **Account Key**: Azure Portal → Storage Account → Access Keys → Key1 or Key2
3. **SAS Token**: Azure Portal → Storage Account → Shared Access Signature → Generate SAS Token

### Memory Issues
- Reduce `MAX_WORKERS` or `BATCH_SIZE`
- Increase `MAX_MEMORY_MB` if you have more RAM

### Processing Errors
- Check logs in `pdf_vectorization.log`
- Some PDFs may be corrupted or password-protected

### Search Quality
- Try different embedding models in configuration
- Adjust chunk size and overlap for better context

## File Structure

```
connection-test/
├── .env                    # Environment variables (configure your credentials here)
├── requirements.txt        # Python dependencies
├── azure_blob_test.py     # Main test script
└── README.md              # This file
```
