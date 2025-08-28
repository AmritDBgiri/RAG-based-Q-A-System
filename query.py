import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join([d.page_content[:1000] for d in docs])

def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        return
    question = sys.argv[1]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using the provided context. If unknown, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(chain.invoke(question))

if __name__ == "__main__":
    main()
