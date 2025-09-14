import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Streamlit Page Setup
st.set_page_config(page_title="🌍 Multilingual Chatbot", page_icon="🤖", layout="centered")

st.title("🌍 Multilingual Chatbot with LangChain + OpenAI")

# API Key Input
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Store Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Type your message in any language..."):
    # Save User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Bot Response
    response = conversation.run(user_input)

    # Save Bot Message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
