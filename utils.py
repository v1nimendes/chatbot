from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

# Nome do modelo
model_name = "gpt-3.5-turbo-0125"

def importacao_documentos(uploaded_files):
    documentos = []
    for uploaded_file in uploaded_files:
        # Salvar o arquivo PDF temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            
        # Usar o caminho do arquivo temporário para o PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
        
        # Remover o arquivo temporário após o carregamento
        os.remove(temp_file_path)
        
    return documentos

def split_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)
    
    for i, doc in enumerate(documentos):
        doc.metadata["source"] = doc.metadata.get("source", f"documento_{i}.pdf")
        doc.metadata["doc_id"] = i
    return documentos

def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store

def cria_chain_conversa(uploaded_files):
    documentos = importacao_documentos(uploaded_files)
    documentos = split_documentos(documentos)
    vector_store = cria_vector_store(documentos)
    
    chat = ChatOpenAI(model=model_name)
    memory = ConversationBufferMemory(return_messages=True,
                                      memory_key="chat_history",
                                      output_key="answer")
    retriever = vector_store.as_retriever()

    prompt_template = """
    Você é um analista de suporte técnico altamente experiente da empresa, responsável por ajudar novos funcionários a esclarecer dúvidas e resolver problemas relacionados aos equipamentos da organização. Seu profundo conhecimento técnico permite que você explique cada situação de maneira detalhada, garantindo que o funcionário compreenda completamente a solução e possa aplicá-la de forma autônoma.

        Como responder:
        Se a pergunta mencionar um equipamento dentro do contexto:

        Pergunte ao funcionário exatamente o que ele deseja saber sobre esse equipamento antes de fornecer uma resposta.
        Se a pergunta for sobre um problema ou dúvida específica:

        Explique o problema de forma clara e objetiva.
        Detalhe um passo a passo preciso para a solução.
        Forneça exemplos ou explicações complementares, se necessário.
        Se a informação não estiver no documento:

        Informe o funcionário de maneira educada e recomende que ele entre em contato com o suporte técnico para obter ajuda adicional.
        Se a pergunta estiver fora do contexto:

        Deixe claro que você só pode responder perguntas relacionadas ao contexto fornecido.
        Contexto da conversa:
        {chat_history}

        Informações relevantes do documento:
        {context}

        Pergunta do funcionário:
        {question}

        Forneça uma resposta detalhada, clara e objetiva, garantindo que o funcionário compreenda completamente a solução. Se a pergunta não estiver dentro do contexto, informe que você só responde perguntas relacionadas ao contexto fornecido.
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
