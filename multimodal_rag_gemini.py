
import base64
import os
import sys
import uuid
import pickle
from pathlib import Path

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    from langchain.retrievers.multi_vector import MultiVectorRetriever


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not found. Add it to your .env file."
    )

CHAT_MODEL = "gemini-3.8-flash"
EMBED_MODEL = "models/gemini-embedding-001"

ID_KEY = "doc_id"
BATCH_SIZE = 2
TOP_K = 4

# A new collection per run prevents stale Chroma entries from
# being matched with an empty in-memory document store.
COLLECTION_NAME = "multimodal_rag_" + uuid.uuid4().hex[:12]


# ============================================================
# GEMINI MODELS
# ============================================================

def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.2,
        max_retries=3,
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
    )


# ============================================================
# IMAGE UTILITIES
# ============================================================

def image_mime_type(data):
    """Identify common image formats from Base64 data."""
    if not isinstance(data, str) or not data.strip():
        return None

    try:
        raw = base64.b64decode(data[:64])
    except Exception:
        return None

    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"

    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"

    return None


def is_base64_image(data):
    return image_mime_type(data) is not None


# ============================================================
# PDF PARTITIONING
# ============================================================

def partition(pdf_path):
    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts = []
    tables = []
    images = []

    for element in elements:
        kind = type(element).__name__

        if kind == "Table":
            html = getattr(element.metadata, "text_as_html", None)

            if isinstance(html, str) and html.strip():
                tables.append(html.strip())

        elif kind == "CompositeElement":
            text = getattr(element, "text", None)

            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

            original_elements = (
                getattr(element.metadata, "orig_elements", None) or []
            )

            for child in original_elements:
                if type(child).__name__ != "Image":
                    continue

                image_data = getattr(
                    child.metadata, "image_base64", None
                )

                if is_base64_image(image_data):
                    images.append(image_data)

        elif kind == "Image":
            image_data = getattr(
                element.metadata, "image_base64", None
            )

            if is_base64_image(image_data):
                images.append(image_data)

    return texts, tables, images


def partition_cached(pdf_path):
    """Cache extraction results so repeated runs do not repartition."""
    cache_path = str(Path(pdf_path).resolve()) + ".cache.pkl"

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as file:
                cached = pickle.load(file)

            if (
                isinstance(cached, tuple)
                and len(cached) == 3
                and all(isinstance(items, list) for items in cached)
            ):
                print("Loaded cached PDF extraction.")
                return cached

        except (OSError, pickle.PickleError, EOFError, ValueError):
            print("Cache is invalid; extracting the PDF again.")

    result = partition(pdf_path)

    with open(cache_path, "wb") as file:
        pickle.dump(result, file)

    return result


# ============================================================
# SUMMARIZATION
# ============================================================

def batch_safely(chain, items):
    """Summarize valid inputs and discard empty results."""
    valid_items = [
        item for item in items
        if isinstance(item, str) and item.strip()
    ]

    if not valid_items:
        return []

    results = chain.batch(
        valid_items,
        config={"max_concurrency": BATCH_SIZE},
    )

    summaries = []

    for item, result in zip(valid_items, results):
        if isinstance(result, str) and result.strip():
            summaries.append(result.strip())
        else:
            # Do not store empty summaries as embeddings.
            print("Warning: Skipped an empty summary.")

    return summaries


def summarize_text_and_tables(texts, tables):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following document content for semantic search. "
        "Preserve important facts, names, dates, figures, keywords, "
        "and relationships. For tables, explain the headers and key values. "
        "Return only the summary.\n\n"
        "Content:\n{element}"
    )

    chain = prompt | get_llm() | StrOutputParser()

    print("Summarizing text...")
    text_summaries = batch_safely(chain, texts)

    print("Summarizing tables...")
    table_summaries = batch_safely(chain, tables)

    # Preserve matching raw content and summaries.
    valid_texts = [
        x for x in texts
        if isinstance(x, str) and x.strip()
    ]
    valid_tables = [
        x for x in tables
        if isinstance(x, str) and x.strip()
    ]

    # Rebuild paired lists to avoid mismatches if a summary fails.
    text_pairs = list(zip(valid_texts, text_summaries))
    table_pairs = list(zip(valid_tables, table_summaries))

    return text_pairs, table_pairs


def summarize_images(images):
    if not images:
        return []

    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            [
                {
                    "type": "text",
                    "text": (
                        "Describe this image for document retrieval. "
                        "Include visible text, labels, chart trends, "
                        "diagrams, relationships, and important details. "
                        "Do not invent information."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:{mime_type};base64,{image}"
                    },
                },
            ],
        )
    ])

    chain = prompt | get_llm() | StrOutputParser()

    valid_images = [
        img for img in images if is_base64_image(img)
    ]

    if not valid_images:
        return []

    inputs = [
        {
            "mime_type": image_mime_type(img),
            "image": img,
        }
        for img in valid_images
    ]

    results = chain.batch(
        inputs,
        config={"max_concurrency": BATCH_SIZE},
    )

    return [
        (img, summary.strip())
        for img, summary in zip(valid_images, results)
        if isinstance(summary, str) and summary.strip()
    ]


# ============================================================
# MULTIVECTOR RETRIEVER
# ============================================================

def build_retriever(text_pairs, table_pairs, image_pairs):
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    docstore = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )

    def add_pairs(pairs):
        # Each pair is (raw content, summary).
        valid_pairs = [
            (raw, summary)
            for raw, summary in pairs
            if isinstance(summary, str)
            and summary.strip()
            and raw is not None
        ]

        if not valid_pairs:
            return

        ids = [str(uuid.uuid4()) for _ in valid_pairs]

        documents = [
            Document(
                page_content=summary.strip(),
                metadata={ID_KEY: doc_id},
            )
            for (_, summary), doc_id in zip(valid_pairs, ids)
        ]

        # Never send empty page_content to the embedding API.
        vectorstore.add_documents(documents)

        # Store the original content behind each summary.
        docstore.mset([
            (doc_id, raw)
            for (raw, _), doc_id in zip(valid_pairs, ids)
        ])

    add_pairs(text_pairs)
    add_pairs(table_pairs)
    add_pairs(image_pairs)

    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            "No searchable summaries were created. "
            "Check the PDF extraction and Gemini summarization output."
        )

    return retriever


# ============================================================
# QUESTION ANSWERING
# ============================================================

def ask(retriever, llm, question):
    question = (question or "").strip()

    if not question:
        print("Please enter a non-empty question.")
        return None, [], []

    print(f"Searching: {question!r}")

    # The original traceback occurred during this call.
    try:
        docs = retriever.invoke(question)
    except Exception as exc:
        print(f"Retrieval/embedding failed: {exc}")
        raise

    raw_items = [
        doc.page_content if isinstance(doc, Document) else doc
        for doc in docs
    ]

    raw_items = [
        item for item in raw_items
        if isinstance(item, str) and item.strip()
    ]

    images = [
        item for item in raw_items
        if is_base64_image(item)
    ]

    texts = [
        item for item in raw_items
        if not is_base64_image(item)
    ]

    context = "\n\n".join(texts)

    content = [
        {
            "type": "text",
            "text": (
                "Answer the question using only the provided PDF context. "
                "If the answer is not present, say that you could not find "
                "it in the document. Distinguish facts from uncertainty.\n\n"
                f"Context:\n{context or '(No text context retrieved.)'}"
                f"\n\nQuestion: {question}"
            ),
        }
    ]

    for img in images:
        mime_type = image_mime_type(img)

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{img}"
            },
        })

    answer = (
        llm | StrOutputParser()
    ).invoke([HumanMessage(content=content)])

    return answer, texts, images


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: python multimodal_rag_gemini.py path/to/file.pdf"
        )

    pdf_path = sys.argv[1]

    try:
        print("Extracting PDF text, tables, and images...")
        texts, tables, images = partition_cached(pdf_path)

        print(
            f"Found {len(texts)} text chunks, "
            f"{len(tables)} tables, {len(images)} images"
        )

        if not texts and not tables and not images:
            sys.exit(
                "No usable content extracted from the PDF. "
                "Check the PDF and Unstructured dependencies."
            )

        print("\nSummarizing extracted content with Gemini...")
        text_pairs, table_pairs = summarize_text_and_tables(
            texts, tables
        )

        image_pairs = []

        if images:
            print("Summarizing images...")
            image_pairs = summarize_images(images)

        print("\nBuilding Chroma retriever...")
        retriever = build_retriever(
            text_pairs, table_pairs, image_pairs
        )

        llm = get_llm()

        print("\nReady. Ask a question (type 'exit' to quit).")

        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                print("Please enter a question.")
                continue

            try:
                answer, source_texts, source_images = ask(
                    retriever, llm, question
                )

                if answer is not None:
                    print(f"\nAnswer:\n{answer}")
                    print(
                        "\nSources: "
                        f"{len(source_texts)} text/table items, "
                        f"{len(source_images)} images"
                    )

            except Exception as exc:
                print(f"\nCould not answer the question: {exc}")
                print(
                    "Check your API key, model availability, "
                    "network connection, and embedding inputs."
                )

    except Exception as exc:
        print(f"\nApplication startup failed: {exc}")
        raise


if __name__ == "__main__":
    main()
