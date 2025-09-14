How to run the Multilingual Chatbot:

1. Install requirements:
   pip install -r requirements.txt

2. Ingest PDFs (sample included in pdfs/):
   python streamlit_langchain_multilingual_chatbot.py ingest pdfs/

3. Start chatbot:
   streamlit run streamlit_langchain_multilingual_chatbot.py

Notes:
- Set OPENAI_API_KEY in your environment if you use OpenAI models.
- After ingestion, FAISS index will be created in faiss_index/.
- Feedback is stored in feedback.json.
