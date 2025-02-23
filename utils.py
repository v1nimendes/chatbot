from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import streamlit as st
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


folder_files = Path(__file__).parent / "files"
model_name = "gpt-3.5-turbo-0125"

def importacao_documentos():
    documentos = []
    for arquivo in folder_files.glob("*.pdf"):
        loader = PyPDFLoader(arquivo)
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos

def split_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)
    
    for i, doc in enumerate(documentos):
        doc.metadata["source"] = doc.metadata["source"].split("/")[-1]
        doc.metadata["doc_id"] = i
    return documentos

def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store

from langchain.prompts import PromptTemplate

def cria_chain_conversa():
    documentos = importacao_documentos()
    documentos = split_documentos(documentos)
    vector_store = cria_vector_store(documentos)
    
    chat = ChatOpenAI(model=model_name)
    memory = ConversationBufferMemory(return_messages=True,
                                      memory_key="chat_history",
                                      output_key="answer")
    retriever = vector_store.as_retriever()

    # Definição do prompt corrigido
    prompt_template = """
    Você é um atendente de suporte técnico de uma empresa. Sua tarefa é auxiliar clientes que possuem dúvidas ou problemas com equipamentos.
    Para cada dúvida apresentada, forneça uma explicação clara e objetiva, detalhando passo a passo o que o cliente precisa fazer para resolver a questão.
    Utilize as informações disponíveis nos documentos fornecidos para embasar suas respostas.

    Se o documento não contiver a informação necessária, diga ao cliente que não encontrou essa informação e recomende que ele entre em contato com o suporte técnico.

    Histórico da conversa:
    {chat_history}

    Informações relevantes do documento:
    {context}

    Pergunta do cliente:
    {question}

    Responda de forma clara e detalhada.
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"], 
        template=prompt_template
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    st.session_state["chain"] = chat_chain
    return chat_chain

