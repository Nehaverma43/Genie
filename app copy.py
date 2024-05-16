import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
google_api_key = "AIzaSyDU7L29UsHIHGhoIICcHYtTvDIQZ4pl5lU"
google_genai_model = "gemini-pro"
pdf_folder_path = r"D:\Chatbot\Pdfs"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key) 
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(google_api_key=google_api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=google_genai_model,
                                   client=genai,
                                   temperature=0.3,
                                   google_api_key=google_api_key,)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def HR_Policies():
    st.session_state.messages = [{"role": "assistant", "content": "Ask your queries related to MSC HR Policies"}]
def MSC_publications():
    st.session_state.messages = [{"role": "assistant", "content": "Ask your queries related to MSC Publications"}]
def KMM_templates():
    st.session_state.messages = [{"role": "assistant", "content": "Ask your queries related to KMM Templates"}]

def clear_history():
    st.session_state.messages = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key) 

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response
def clear_text():
    st.session_state.my_text = st.session_state.widget
    st.session_state.widget = ""

def main():
    st.set_page_config(
        page_title="MSC-Genie",
        page_icon="🤖"
    )

    # Check if the folder exists
    if os.path.isdir(pdf_folder_path):
        pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
       #st.success("PDF files processed successfully")
    else:
        st.error("Invalid folder path!")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(r"D:\Chatbot\Intro.png",width=300)

    with col3:
        st.write(' ')

    # Main content area for displaying chat messages
    #st.image(r"D:\Chatbot\Intro.png",width=300)
    st.markdown('<div style="background-color:#FFA500;padding:10px;border-radius:10px;">'
            '<h2 style="color:#214761;text-align:center;font-size:20px;">Hi there! I am MSC Genie. I am a GenAI Bot that will be answering your queries. Currently, the following repositories are covered.</h2>'
            '</div>', unsafe_allow_html=True)

    st.sidebar.button('Clear Chat History', on_click=clear_history, key="clear_button")

    # Layout for buttons in three columns
    col1, col2, col3 = st.columns(3)  

    with col1:
        if st.button('MSC Publications', on_click=MSC_publications, key="MSC_Publications"):
            MSC_publications()

    with col2:
        if st.button('HR Policies', on_click=HR_Policies, key="HR_Policies"):
            HR_Policies()
            
    with col3:
        if st.button("KMM Templates", on_click=KMM_templates, key="KMM_Templates"):
            KMM_templates()

    # Chat input and message display
    if "messages" not in st.session_state:
        st.session_state.messages = []
   

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.image("user.png", width=50)
            st.markdown(
            f'<div class="st-emotion-cache-4oy321" style="background-color: #D3E0EA; padding: 10px; border-radius: 10px; text-align: left;">{message["content"]}</div>',
           unsafe_allow_html=True
       )
        elif message["role"] == "assistant":
            col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)

            with col1:
                st.write(' ')

            with col2:
                st.write(' ')
            with col3:
                st.write(' ')
            with col4:
                st.write(' ')
            with col5:
                st.write(' ')
            with col6:
                st.write(' ')
            with col7:
                st.write(' ')

            with col8:
                st.image(r"D:\Chatbot\Found.png", width=75)
            #st.image("D:\Chatbot\Found.png", width=50)
            st.markdown(
            f'<div class="message-container" style="display: flex; justify-content: flex-end;">'
            f'<div class="st-emotion-cache-4oy321" style="background-color: #C4E4F7; padding: 10px; border-radius: 10px; text-align: right;">{message["content"]}</div>'
            
            f'</div>',
            unsafe_allow_html=True
        )

    # User input prompt at the end
    with st.form(key='my_form'):
        st.write("Enter your query:")       
        user_query = st.text_input("", key="user_query")        
        st.form_submit_button(label='Submit')
        st.empty()
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            response = user_input(user_query)
            if response is not None:
                if isinstance(response['output_text'], list):
                    for response_item in response['output_text']:
                        st.session_state.messages.append({"role": "assistant", "content": response_item})
                 
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                    
if __name__ == "__main__":
    main()