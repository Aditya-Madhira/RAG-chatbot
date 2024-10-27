import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os

# API keys
GROQ_API_KEY = "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"
HUGGINGFACE_TOKEN = "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"

os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def process_pdf(pdf_file):
    """Process uploaded PDF file."""
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)

    # Clean up
    os.remove("temp.pdf")

    return chunks


@st.cache_resource
def setup_qa_chain(uploaded_file=None):
    """Set up the QA chain."""
    # Create LLM
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile"
    )

    if uploaded_file:
        # Process PDF and set up retrieval chain
        chunks = process_pdf(uploaded_file)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        # Create PDF-specific prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant chatbot. Your goal is to engage in natural conversation and help users with their questions.
            For questions about the uploaded document, use the provided context to give accurate answers: {context}
            For general questions, feel free to draw upon your knowledge to provide helpful responses.
            Always maintain a conversational, friendly tone."""),
            ("human", "{question}")
        ])

        # Create chain with retriever
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
    else:
        # Create general conversation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant chatbot. Your goal is to engage in natural conversation and help users with their questions.
            Draw upon your knowledge to provide helpful responses while maintaining a conversational, friendly tone."""),
            ("human", "{question}")
        ])

        # Create simple chain without retriever
        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

    return chain


def main():563














1222222222220






































































































































































































































































































































































































































































































































































































0    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 PDF Chatbot")

    # File upload (optional)
    uploaded_file = st.file_uploader("Upload a PDF (optional)", type=['pdf'])

    # Initialize chat chain
    if 'chain' not in st.session_state or (uploaded_file and 'last_file' not in st.session_state) or \
            (uploaded_file and st.session_state.get('last_file') != uploaded_file.name):
        with st.spinner("Setting up the chatbot..." if not uploaded_file else "Processing PDF..."):
            st.session_state.chain = setup_qa_chain(uploaded_file)
            if uploaded_file:
                st.session_state.last_file = uploaded_file.name
            st.success("Ready to chat!")

    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(prompt)
                st.write(response)

        # Add AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()