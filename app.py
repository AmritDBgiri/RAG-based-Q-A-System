import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def load_index(index_dir="index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # allow_dangerous_deserialization is required in some environments for FAISS
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db

def format_docs(docs):
    # Limit to reasonable size to avoid token explosion
    formatted = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        src_str = f"{src}" if page is None else f"{src} (page {page})"
        content = d.page_content.strip().replace("\n", " ")
        content = content[:1000]  # trim
        formatted.append(f"[{i}] Source: {src_str}\n{content}")
    return "\n\n".join(formatted)

def build_chain(retriever):
    system = """You are a helpful AI assistant. Use the provided context to answer the user's question.
If the context is insufficient, say you don't know. Be concise and, when possible, cite source numbers like [1], [2]."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Q&A (LangChain + FAISS)", page_icon="🔎")
    st.title("🔎 RAG-based Q&A System (LangChain + FAISS + OpenAI)")

    with st.sidebar:
        st.header("Index")
        st.write("Make sure you've run `python ingest.py` after adding files to the `data/` folder.")
        k = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=4, step=1)
        st.markdown("---")
        st.caption("Model: gpt-4o-mini | Embeddings: text-embedding-3-small")

    try:
        db = load_index()
    except Exception as e:
        st.error(f"Failed to load FAISS index. Did you run `python ingest.py`? Error: {e}")
        return

    retriever = db.as_retriever(search_kwargs={"k": k})
    chain = build_chain(retriever)

    question = st.text_input("Ask a question about your documents:")
    show_ctx = st.checkbox("Show retrieved context", value=False)

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            answer = chain.invoke(question.strip())
        st.subheader("Answer")
        st.write(answer)

        if show_ctx:
            with st.expander("Retrieved context"):
                docs = retriever.get_relevant_documents(question.strip())
                st.code(format_docs(docs))

if __name__ == "__main__":
    main()
