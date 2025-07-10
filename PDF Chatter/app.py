import streamlit as st
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
def get_conversation_chain(vectorizer):
    llm = OpenAI(temperature=0.0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorizer.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

def get_vectorizer(text_chunks):
    #embeddings=OpenAIEmbeddings()
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    except Exception as e:
        print("Error loading embeddings:", e)
    vectorizer = FAISS.from_texts(text_chunks, embeddings=embeddings)
    return vectorizer

def get_pdf_texts(uploaded_files):
    from PyPDF2 import PdfReader
    raw_texts = []
    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        raw_texts.append(text)
    return raw_texts

def get_text_chunks(raw_texts, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    text_chunks = []
    for text in raw_texts:
        chunks = text_splitter.split_text(text)
        text_chunks.extend(chunks)
    return text_chunks

def handle_userinput(user_question):
    if "conversation" in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.conversation.memory.add_user_message(user_question)
        st.session_state.conversation.memory.add_ai_message(response["answer"])
        st.write(bot_template.replace("{{MSG}}", response["answer"]), unsafe_allow_html=True)
    else:
        st.error("Please process your PDFs first.")
    st.session_state.question = ""


def main():
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    #st.set_page_config(page_title="PDF Chatter", page_icon=":book:")
    st.header("Chat with multiple PDFs")
    user_question=st.text_input("Ask a question about your PDFs:", key="question")
    with st.sidebar:
        st.subheader("Upload your PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                st.write(f"Uploaded: {file.name}")
        if st.button("Process PDFs", key="process_pdfs"):
            with st.spinner("Processing PDFs..."):
                raw_texts = get_pdf_texts(uploaded_files)
                text_chunks=get_text_chunks(raw_texts)
                vectorizer=get_vectorizer(text_chunks)
                st.session_state.conversation=get_conversation_chain(vectorizer)
                st.success("PDFs processed successfully!")
    if user_question:
        handle_userinput(user_question)
#st.session_state.conversation

if __name__ == "__main__":
    main()  