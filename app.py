import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history  import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
#code

from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

##set up streamlit
st.title("Coversational RAG with PDF uploads and chat history")
st.write("Upload PDF's and chat with their content")

##Input the GRoq API KEY
api_key=st.text_input("Enter your GRoq API KEY:", type="password")

##check if GRoq API KEY is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    #chat interface
    session_id=st.text_input("Session_id", value="default_session")
    # statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    #Process Uploaded Pdfs
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open (temppdf,"wb") as file:
                file.write (uploaded_files.getvalue())
                file_name=uploaded_files.name
            loader=PyPDFLoader(temppdf)    
            docs=loader.load()
            documents.extend(docs)

#split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might refer context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do not answer the question"
            "just formulate it if needed and otherwise return as it is"
        )
        contextualize_q_Prompt= ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_Prompt)
    #Answer Question

        system_prompt=(
            "You are an assistant for question-answering tasks"
            "use the following pieces of retrieved contexts to answer"
            "If you dont know the answer, say that you dont know"
            "Use three sentences maximum and keep the answer concise" \
            "{context}"
        )
        qa_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}"),
                ]
            )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input=st.text_input("Your Question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the GRoq API KEY")
        


