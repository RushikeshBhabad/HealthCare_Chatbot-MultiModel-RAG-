import streamlit as st
from ingest import ingest_pdf
from rag import run_rag

st.set_page_config(
    page_title="Healthcare Multimodal RAG",
    layout="wide"
)

st.title("🩺 Healthcare Multimodal RAG Chatbot")

@st.cache_resource
def load_data():
    vectorstore, image_store = ingest_pdf(
        "data/disease-handbook-complete.pdf"
    )
    return vectorstore, image_store


vectorstore, image_store = load_data()

query = st.text_input("Ask a healthcare question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = run_rag(query, vectorstore, image_store)
        st.markdown("### Answer")
        st.write(answer)
