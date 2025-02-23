import streamlit as st
from utils import cria_chain_conversa
import os

# Certifique-se de que a chave da API OpenAI esteja configurada
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("A chave da API OpenAI não está configurada. Verifique as variáveis de ambiente.")
    st.stop()

def chat_window():
    st.header("SensorChat", divider=True)
    if 'chain' not in st.session_state:
        st.error("Faça o upload de PDFs para começar")
        st.stop()
    
    chain = st.session_state["chain"]
    memory = chain.memory 

    mensagens = memory.load_memory_variables({})["chat_history"]
    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)
        
    nova_mensagem = st.chat_input("Converse com seus documentos")
    if nova_mensagem:
        chat = container.chat_message("human")
        chat.markdown(nova_mensagem)
        chat = container.chat_message("ai")
        chat.markdown("Gerando Resposta")
        resposta = chain({"question": nova_mensagem})
        chat.markdown(resposta['answer'])  # Corrigido para mostrar a resposta

def main():
    with st.sidebar:
        st.header("Upload de PDFs")
        uploaded_pdfs = st.file_uploader("Adicione arquivos PDF", 
                                         type="pdf", 
                                         accept_multiple_files=True)
        if uploaded_pdfs:
            st.session_state['uploaded_pdfs'] = uploaded_pdfs
            st.success(f"{len(uploaded_pdfs)} arquivo(s) salvo(s) com sucesso!")
        
        label_botao = "Inicializar Chatbot"
        if "chain" in st.session_state:
            label_botao = "Atualizar Chatbot"
        if st.button(label_botao, use_container_width=True):
            if 'uploaded_pdfs' not in st.session_state or len(st.session_state['uploaded_pdfs']) == 0:
                st.error("Adicione arquivos pdf para inicializar o chatbot")
            else:
                st.success("Inicializando o Chatbot...")
                cria_chain_conversa(st.session_state['uploaded_pdfs'])
                st.experimental_rerun()
    chat_window()

if __name__ == "__main__":
    main()
