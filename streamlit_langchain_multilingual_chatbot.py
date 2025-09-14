import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Streamlit page setup
st.set_page_config(page_title="🌍 Multilingual Chatbot", page_icon="🤖", layout="centered")

st.title("🌍 Multilingual Chatbot with LangChain + OpenAI")

# Get OpenAI API Key
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key to start.")
    st.stop()

# Initialize chatbot
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message in any language..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    response = conversation.run(user_input)

    # Display bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
