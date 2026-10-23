# 📚 Multimodal RAG: Chat with PDFs (Images, Tables & Text)

A end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** pipeline that allows you to chat with complex PDF documents containing text, embedded tables, and visual diagrams/charts.

This project extracts text, HTML tables, and Base64-encoded image payloads using `unstructured`, generates element-specific summaries, stores them using LangChain's `MultiVectorRetriever` with ChromaDB, and passes visual + text context to a Vision-capable LLM (`gpt-4o-mini`) for precise QA.

---

## 🏗️ Architecture Overview

1. **Document Partitioning (`unstructured`):** Extracts raw text, tables (as HTML), and visual blocks (as Base64 payloads) using a high-resolution `by_title` strategy.
2. **AI Summarization:** 
   - **Text & Tables:** Summarized using `Llama-3.1` (via Groq) for high-density semantic indexing.
   - **Images & Diagrams:** Summarized/described using `GPT-4o-mini` (Vision capabilities).
3. **Multi-Vector Storage:**
   - **Vector Store (ChromaDB):** Stores vector embeddings of the **summaries** for optimal similarity search.
   - **Document Store (In-Memory / Key-Value):** Holds original raw text chunks, HTML tables, and Base64 image payloads linked by a unique `doc_id`.
4. **Multimodal QA:** Retrieves relevant assets and feeds both textual context and raw visual data into a multimodal LLM to synthesize final answers.

---

## 🛠️ Prerequisites & Installation (Windows Guide)

### 1. Install System Binaries
The PDF partitioning engine requires external native binaries on Windows:

* **Tesseract OCR:** 
  1. Download the installer from the [UB-Mannheim Tesseract GitHub Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
  2. Install to `C:\Program Files\Tesseract-OCR`.
  3. Add `C:\Program Files\Tesseract-OCR` to your Windows System `PATH`.

* **Poppler for Windows:**
  1. Download the latest precompiled release zip from [Poppler Releases](https://github.com/oschwartz10612/poppler-windows/releases/).
  2. Extract to `C:\Program Files\poppler`.
  3. Add `C:\Program Files\poppler\Library\bin` (or `\bin`) to your Windows System `PATH`.

---

### 2. Python Setup

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/YOUR_USERNAME/multimodal-pdf-rag.git](https://github.com/YOUR_USERNAME/multimodal-pdf-rag.git)
cd multimodal-pdf-rag

python -m venv venv
venv\Scripts\activate