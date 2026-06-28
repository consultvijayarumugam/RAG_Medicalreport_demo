import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import tempfile

# ====================== UI SETUP ======================
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
    .header { text-align: center; padding: 2.5rem; 
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; border-radius: 15px; margin-bottom: 2rem; }
    .section { background: white; padding: 2rem; border-radius: 15px; 
               box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
    .answer-box { background: #f0f8ff; padding: 1.8rem; border-radius: 15px; 
                  border-left: 6px solid #667eea; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🩺 Medical Report Assistant</h1><p>Intelligent Q&A for your father\'s medical report</p></div>', unsafe_allow_html=True)

# Check API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

# Session State
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# Sidebar
with st.sidebar:
    st.success("✅ API Key Loaded")
    st.info("RAG App using LangChain + FAISS")
    st.caption("Assignment 6 - Gen AI Architect")

# ====================== UPLOAD SECTION ======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📄 Upload Medical Report")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.vectorstore = None
        st.session_state.pdf_processed = False
        st.session_state.file_name = uploaded_file.name

    st.success(f"✅ Uploaded: **{uploaded_file.name}**")

    if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
        with st.spinner("Processing PDF and creating vector database..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings()
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

                st.session_state.pdf_processed = True
                st.success(f"✅ Vector Database Ready! ({len(chunks)} chunks)")

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

st.markdown('</div>', unsafe_allow_html=True)

# ====================== QUERY SECTION ======================
if st.session_state.pdf_processed:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("💬 Ask Questions")

    query = st.text_input("Your Question:", 
                         placeholder="What are the key findings in the report?")

    if st.button("🔍 Get Answer", type="primary", use_container_width=True) and query:
        with st.spinner("Searching report and generating answer..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(query, k=4)
                context = "\n\n".join([d.page_content for d in docs])

                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

                prompt = f"""Answer the question based only on the following context:

Context:
{context}

Question: {query}

Answer:"""

                response = llm.invoke(prompt)
                answer = response.content

                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 📋 Answer")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📚 Source Excerpts from Report"):
                    for i, doc in enumerate(docs):
                        st.caption(f"Source {i+1}")
                        st.write(doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Upload your PDF and click **Process & Index Document** to begin.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666'>Your father's medical data is processed securely and not stored.</p>", unsafe_allow_html=True)