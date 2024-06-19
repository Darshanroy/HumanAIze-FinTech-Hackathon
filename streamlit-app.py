import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import UnstructuredCSVLoader,PyPDFDirectoryLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import HypotheticalDocumentEmbedder

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit app title
st.title("LAW QA System with RAG and HYDE")

# Initialize embeddings and models
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
)

mistral_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=3000,
    do_sample=False,
)

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    mistral_llm,
    embeddings,
    prompt_key="web_search"
)

# Define the chat prompt template for QA
qa_prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}
""")

def prepare_vector_store(hyde_embeddings):
    """Prepares the vector store database by loading documents and creating embeddings."""
    st.session_state.embeddings = hyde_embeddings
    if "vector_store" not in st.session_state:
        file_path = "./chroma"
        if os.path.exists(file_path):
            st.session_state.vector_store = Chroma(persist_directory="./chroma", embedding_function=hyde_embeddings)
        else: 
            loader = PyPDFDirectoryLoader(r"crime data\murder")
            st.session_state.docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=330, chunk_overlap=20)
            final_documents = text_splitter.split_documents(st.session_state.docs)
            st.session_state.vector_store = Chroma.from_documents(final_documents[:10000], hyde_embeddings,persist_directory="./chroma")

# Define the contextualization prompt for reformulating questions based on chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Statefully manage chat history
chat_history_store = {}

def get_chat_session_history(session_id: str) -> BaseChatMessageHistory:
    """Fetches the chat history for the given session."""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

def process_question(user_question, session_id, llm, vector_store, qa_prompt_template, contextualize_q_prompt):
    """
    Processes the user question using the given LLM and vector store.
    
    Args:
    - user_question (str): The user's question.
    - session_id (str): The session ID for chat history.
    - llm (HuggingFaceEndpoint): The LLM to use for generating answers.
    - vector_store (Chroma): The vector store for retrieving documents.
    - qa_prompt_template (ChatPromptTemplate): The template for generating answers.
    - contextualize_q_prompt (ChatPromptTemplate): The template for contextualizing questions.
    
    Returns:
    - response (dict): The generated response and associated context documents.
    - response_time (float): The time taken to generate the response.
    """
    if user_question == 'q':
        return None, 0
    
    # Create the question answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)

    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create the conversational RAG chain with chat history management
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_chat_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Measure response time and invoke the RAG chain
    start_time = time.process_time()
    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}},
    )
    response_time = time.process_time() - start_time

    return response, response_time

# Button to initialize document embedding
if st.button("Initialize Document Embedding"):
    prepare_vector_store(hyde_embeddings)
    st.write("Vector store database is ready")

# User input for the question
user_question = st.text_input("Enter your question based on the documents")

# Button to process the question
if st.button("Submit Question"):
    response, response_time = process_question(
        user_question,
        session_id="abc123",
        llm=mistral_llm,  # Replace with the desired LLM (e.g., Llama3LLM)
        vector_store=st.session_state.vector_store,
        qa_prompt_template=qa_prompt_template,
        contextualize_q_prompt=contextualize_q_prompt
    )

    if response:
        st.write(f"Response time: {response_time} seconds")
        st.write(response['answer'])

        # Display document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
