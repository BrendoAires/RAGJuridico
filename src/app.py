import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_sistema_rag():
    if not os.path.exists("chroma_db"):
        raise FileNotFoundError("O banco vetorial 'chroma_db' não existe. Execute primeiro o script de ingestão.")

    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    vector_db = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash",
        temperature=0.2
    )

    system_prompt = (
        "Você é um assistente jurídico sênior especializado em análise processual.\n"
        "Sua tarefa é analisar os documentos do processo fornecidos abaixo e responder à pergunta do advogado "
        "com extrema precisão, objetividade e clareza.\n\n"
        "REGRAS DE RESPOSTA:\n"
        "1. Responda APENAS com base nos trechos fornecidos no contexto.\n"
        "2. Se a informação não estiver presente nos documentos, diga claramente: 'Informação não encontrada no processo analisado.'\n"
        "3. Sempre cite o nome do documento fonte de onde extraiu a resposta (ex: Conforme a Petição Inicial...).\n\n"
        "Contexto dos Documentos:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Construção moderna da Chain usando LCEL (dispensa create_retrieval_chain)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def perguntar_ao_processo(pergunta: str):
    chain, retriever = inicializar_sistema_rag()
    
    # Executa a busca
    resposta = chain.invoke(pergunta)
    # Busca os documentos na mão apenas para exibir as fontes na tela
    documentos_fontes = retriever.invoke(pergunta)
    
    print("\n================ RESPOSTA DA IA ================")
    print(resposta)
    print("\n================ FONTES CONSULTADAS ================")
    for idx, doc in enumerate(documentos_fontes):
        fonte = doc.metadata.get("source", "Documento desconhecido")
        print(f"[{idx+1}] Arquivo: {fonte}")
    print("===================================================\n")

if __name__ == "__main__":
    pergunta_teste = "Quais são os principais argumentos apresentados pela defesa ou pela acusação?"
    print(f"Perguntando: '{pergunta_teste}'...")
    perguntar_ao_processo(pergunta_teste)
