import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from pipeline import MongoDBConnection, EmbeddingModelSentence, ModelResponse
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv("api.env")

# Unique session ID
session_id = str(uuid.uuid4())

st.set_page_config(page_title="Steam Game Recommender", page_icon="🎮")
st.title("🎮 Ask me for Any steam game!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load models and DB connection once
@st.cache_resource
def load_resources():
    mongo_conn = MongoDBConnection(mongo_access=os.environ.get("mongodb_access"))  # from .env
    embedding_model = EmbeddingModelSentence(SentenceTransformer("all-MiniLM-L6-v2"))
    model_response = ModelResponse(gemini_api_key=os.environ.get("gemini_api_key"))
    return mongo_conn, embedding_model, model_response

mongo_conn, embedding_model, model_response = load_resources()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("What kind of games are you looking for?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text = model_response.process_response(
                    user_query=prompt,
                    collection=mongo_conn.collection,
                    embedding_model=embedding_model
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"❌ Error: {e}")
