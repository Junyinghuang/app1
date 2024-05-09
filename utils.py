from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def qa_agent(openai_api_key, memory, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key,
                   temperature=0)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                    openai_api_key=openai_api_key)
    db = FAISS.load_local("./faissdb", embeddings_model,allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type="map_reduce"
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
