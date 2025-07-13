import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Streamlit app title
st.title("RAG Q&A Chatbot for 'If—' by Rudyard Kipling")
st.subheader("Ask questions about the poem!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load and process the document
@st.cache_resource
def load_and_index_document():
    # Load the text file
    file_path = "if_by_rudyard_kipling.txt"
    if not os.path.exists(file_path):
        st.error("Poem file 'if_by_rudyard_kipling.txt' not found. Please ensure it's in the app directory.")
        return None
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1500,  # Increased to avoid chunk size warning
        chunk_overlap=50,
        separator="\n"  # Split on newlines for poem structure
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# Set up the LLM
@st.cache_resource
def load_llm():
    model_id = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # GPU for Colab, CPU for Streamlit Cloud
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

# Create RAG pipeline
@st.cache_resource
def create_rag_chain():
    vectorstore = load_and_index_document()
    if vectorstore is None:
        return None
    llm = load_llm()
    
    prompt_template = """Use the following context to answer the question. If the answer is not in the context, say so. Keep the answer concise and under three sentences. Always end with "Thanks for asking!"

Context: {context}

Question: {question}

Answer: """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt}
    )

# Load RAG chain
rag_chain = create_rag_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about the poem:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if rag_chain is None:
                st.error("Cannot process question due to missing poem file.")
            else:
                response = rag_chain.run(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
