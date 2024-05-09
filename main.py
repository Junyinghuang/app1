import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key：", type="password")
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

question = st.text_input("Your question")

if question and not openai_api_key:
    st.info("Please provide your OpenAI API key")

if question and openai_api_key:
    with st.spinner("One moment, AI is thinking..."):
        response = qa_agent(openai_api_key, st.session_state["memory"], question)
    st.write("### Answer")
    st.write(response["answer"])
    for i in range(3):
        st.write(str(response["source_documents"][i].metadata))
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("History"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
