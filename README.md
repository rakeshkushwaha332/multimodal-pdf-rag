# 📚 Multimodal RAG: Chat with PDFs (Images, Tables & Text)

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with complex PDF documents containing text, tables, images, diagrams, and charts.

The project extracts text, HTML tables, and Base64-encoded image payloads using `unstructured`, generates element-specific summaries, and indexes them using LangChain's `MultiVectorRetriever` and ChromaDB. During question answering, relevant textual and visual context is passed to a vision-capable LLM (`gpt-4o-mini`) to generate context-aware responses.

---

## 🏗️ Architecture Overview
<img width="1024" height="317" alt="image" src="https://github.com/user-attachments/assets/fd6e8ec6-5314-4912-8fcd-b35fa7313461" />

### 1. Document Partitioning
**Technology:** `unstructured`

- Processes PDFs using a high-resolution partitioning strategy.
- Extracts text elements, tables as HTML, and visual elements as Base64-encoded image payloads.
- Uses the `by_title` chunking strategy to organize document content into meaningful sections.

### 2. AI-Powered Summarization

- **Text and tables:** Uses Llama 3.1 through Groq to generate concise summaries for semantic indexing.
- **Images and diagrams:** Uses GPT-4o-mini's vision capabilities to describe visual content.

### 3. Multi-Vector Storage

- **Vector store — ChromaDB:** Stores embeddings of element summaries for efficient similarity-based retrieval.
- **Document store — In-memory key-value store:** Maintains the original text chunks, HTML tables, and Base64-encoded image payloads.
- **Document linking:** Associates each summary with its original content using a unique `doc_id`.

### 4. Multimodal Question Answering

1. Accepts a user question about an uploaded or processed PDF.
2. Retrieves relevant document elements using semantic similarity search.
3. Loads the original text, tables, and visual assets associated with the retrieved summaries.
4. Passes the relevant context to a multimodal LLM.
5. Generates an answer grounded in the retrieved document content.

---

## 🛠️ Prerequisites

Before setting up the project, ensure you have the following installed:

- Python
- Git
- Tesseract OCR
- Poppler for Windows
- API keys for the LLM providers used by the project

### 1. Install System Dependencies (Windows)

#### Tesseract OCR

Tesseract OCR is used for optical character recognition when processing scanned documents and images.

1. Download the installer from the [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install Tesseract to:

   ```text
   C:\Program Files\Tesseract-OCR
   ```

3. Add the installation directory to your Windows `PATH` environment variable.

#### Poppler

Poppler provides PDF utilities required by several PDF-processing workflows.

1. Download a precompiled Windows release from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
2. Extract the archive to a directory such as:

   ```text
   C:\Program Files\poppler
   ```

3. Add the directory containing the Poppler executables to your Windows `PATH`. Depending on the downloaded release, this may be:

   ```text
   C:\Program Files\poppler\Library\bin
   ```

   Verify the actual directory structure after extraction.

4. Open a new terminal and verify the installations:

   ```powershell
   tesseract --version
   pdfinfo -v
   ```

---

## 🚀 Installation and Setup

### 1. Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-pdf-rag.git
cd multimodal-pdf-rag
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

If PowerShell blocks script execution, use Command Prompt or adjust your execution policy according to your organization's security requirements.

### 4. Install Python Dependencies

If the repository contains a `requirements.txt` file, install the dependencies with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency file is not yet available, install the packages required by your implementation and document their versions before running the project.

### 5. Configure Environment Variables

Create a `.env` file in the project root and add the API keys required by your implementation.

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

Use the appropriate API keys for the providers configured in your code. Never commit your actual API keys or other secrets to GitHub.

Add the following entries to `.gitignore` if they are not already present:

```gitignore
.env
venv/
__pycache__/
*.py[cod]
.chroma/
```

---

## 🔍 Key Technologies

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Unstructured | PDF partitioning and content extraction |
| LangChain | Retrieval and LLM orchestration |
| MultiVectorRetriever | Connects indexed summaries with original content |
| ChromaDB | Vector storage and semantic search |
| Groq / Llama 3.1 | Text and table summarization |
| OpenAI GPT-4o-mini | Visual understanding and multimodal question answering |
| Tesseract OCR | Optical character recognition |
| Poppler | PDF processing utilities |

---

## 💡 Key Features

- **Multimodal PDF processing:** Works with text, tables, diagrams, and images.
- **Summary-based retrieval:** Indexes concise summaries to improve semantic search.
- **Original-content retrieval:** Links summaries to the underlying document elements.
- **Vision-enabled question answering:** Uses visual context when answering questions about charts, diagrams, and other images.
- **Modular architecture:** Separates extraction, summarization, indexing, retrieval, and answer generation.

---

## ⚙️ How It Works

```text
PDF Document
     |
     v
Unstructured PDF Partitioning
     |
     +------ Text Chunks
     |
     +------ HTML Tables
     |
     +------ Images / Diagrams
                |
                v
       Element Summarization
                |
                v
        Summary Embeddings
                |
                v
          ChromaDB Index
                |
                v
         Semantic Retrieval
                |
                v
    Original Content Retrieval
                |
                v
      Multimodal LLM (GPT-4o-mini)
                |
                v
          Answer to User
```

---

## 🔐 Security Notes

- Store API keys in environment variables or a secrets manager.
- Never commit `.env` files containing credentials.
- Avoid logging API keys or sensitive document contents.
- Review the privacy requirements of the LLM providers before submitting confidential PDFs.

---

## 📌 Future Improvements

- Add a web-based chat interface.
- Support multiple PDF uploads and document collections.
- Add persistent document storage instead of relying exclusively on in-memory storage.
- Improve retrieval using hybrid search and reranking.
- Add source-page references and citations to generated answers.
- Add automated tests for extraction, retrieval, and question answering.

---

## 📄 License

Add a license file if you intend to distribute this project publicly. Choose a license that matches your intended usage and distribution requirements.
