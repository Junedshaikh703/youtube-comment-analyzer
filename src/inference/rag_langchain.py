from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 🔹 Create vector store from transcript
def build_vector_store(transcript: str):

    print("\n[RAG] 🔧 Building vector store...")

    # 1️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(transcript)

    print(f"[RAG] Chunks created: {len(chunks)}")
    print(f"[RAG] Sample chunk: {chunks[0][:100] if chunks else 'None'}")

    # 2️⃣ Embeddings
    print("[RAG] Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3️⃣ Vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    print("[RAG] ✅ Vector store ready")

    return vector_store


# 🔹 Retrieve relevant chunks
def retrieve_context(query: str, vector_store, k: int = 3):

    print("\n[RAG] 🔍 Retrieving context...")
    print(f"[RAG] Query: {query[:80]}")

    docs = vector_store.similarity_search(query, k=k)

    print(f"[RAG] Retrieved chunks: {len(docs)}")

    context = " ".join([doc.page_content for doc in docs])

    print(f"[RAG] Context preview: {context[:150]}")

    return context