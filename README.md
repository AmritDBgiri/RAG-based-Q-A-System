# 🔎 RAG-based Q&A System (LangChain + FAISS + OpenAI)

This project is a **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **FAISS**, and **OpenAI**.  

It allows you to chat with your **local documents** (`.pdf`, `.txt`) by:
- 📂 Running a one-time **ingestion script** to index documents  
- 💬 Asking questions via a simple **Streamlit UI**  
- 💻 Using a **CLI tool** for quick terminal queries  

✅ Tested with **Python 3.10 / 3.11** on Linux & Windows.

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables template
cp .env.example .env
# Then open .env and paste your OpenAI API key
👉 Don’t have an OpenAI key?
You can adapt the code to use a local model (e.g., Ollama / Llama.cpp) by swapping ChatOpenAI for a local LLM wrapper.

📂 Add Documents
Place your source files in the data/ folder. Supported formats:

📑 .pdf (via PyPDFLoader)

📄 .txt (via TextLoader)

An example file data/sample.txt is already included.

🏗️ Build the Vector Index (FAISS)
bash
Copy code
python ingest.py
This will create/update the FAISS index under index/ using text-embedding-3-small.

💬 Run the Q&A App (Streamlit)
bash
Copy code
streamlit run app.py
Open the local URL shown in the terminal (e.g., http://localhost:8501)

Ask questions about your docs

Optionally toggle “Show retrieved chunks” to inspect retrievals

⚡ Quick CLI Query (Optional)
bash
Copy code
python query.py "What is this project about?"
📂 Project Structure
graphql
Copy code
rag_faiss_langchain/
├─ app.py           # Streamlit Q&A app
├─ ingest.py        # Build/update FAISS index
├─ query.py         # CLI for quick queries
├─ requirements.txt
├─ .env.example     # Template for secrets
├─ data/            # Your PDFs/TXT here
│  └─ sample.txt
└─ index/           # Auto-generated FAISS index
📝 Notes & Tips
Use the same embedding model for both indexing & querying

Tune chunk_size and chunk_overlap in ingest.py

Adjust retriever k (number of top chunks) in app.py

Customize prompts in app.py for domain-specific Q&A

🧪 Evaluation Ideas
Create a small YAML/JSON of Q&A pairs and compare answers (semantic overlap / keyword match)

Track latency and retrieval relevance

🛠️ Troubleshooting
If FAISS throws a warning → keep allow_dangerous_deserialization=True in app.py

For complex PDFs, consider preprocessing or using Unstructured loaders

✨ This project is designed for quick onboarding into RAG pipelines and showcases skills in:

Agentic AI workflows

LangChain orchestration

FAISS vector search

Prompt engineering & debugging








