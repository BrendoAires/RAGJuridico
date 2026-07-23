import os
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importa a função de inicialização do app.py
from src.app import inicializar_sistema_rag

# Instância do FastAPI
app = FastAPI(
    title="LegalMind Analytics API",
    description="API para análise preditiva e consulta RAG em documentos processuais.",
    version="1.0.0"
)

# Configuração de CORS (permite integração com qualquer frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validação de entrada e saída
class PerguntaRequest(BaseModel):
    pergunta: str = Field(
        ..., 
        example="Quais são os principais argumentos apresentados pela defesa?",
        description="A pergunta a ser respondida com base no processo."
    )

class FonteResponse(BaseModel):
    arquivo: str
    conteudo: str

class RespostaResponse(BaseModel):
    pergunta: str
    resposta: str
    fontes: List[FonteResponse]

# Inicializa o motor RAG uma única vez durante a inicialização da API
rag_chain = None
retriever = None

@app.on_event("startup")
def startup_event():
    global rag_chain, retriever
    try:
        rag_chain, retriever = inicializar_sistema_rag()
        print("=== Motor RAG inicializado com sucesso na API ===")
    except Exception as e:
        print(f"Erro ao carregar banco vetorial no startup: {e}")

@app.get("/", tags=["Healthcheck"])
def health_check():
    return {"status": "online", "service": "LegalMind Analytics RAG Engine"}

@app.post("/api/v1/consultar", response_model=RespostaResponse, tags=["RAG"])
def consultar_processo(payload: PerguntaRequest):
    if not rag_chain or not retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O serviço RAG não foi inicializado corretamente. Verifique se o 'chroma_db' foi criado."
        )

    try:
        # Executa a geração da resposta via LCEL
        resposta_txt = rag_chain.invoke(payload.pergunta)
        
        # Recupera os documentos originais citados
        documentos = retriever.invoke(payload.pergunta)
        
        fontes = [
            FonteResponse(
                arquivo=doc.metadata.get("source", "Documento desconhecido"),
                conteudo=doc.page_content[:300] + "..." # Trecho inicial do chunk
            )
            for doc in documentos
        ]

        return RespostaResponse(
            pergunta=payload.pergunta,
            resposta=resposta_txt,
            fontes=fontes
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a consulta: {str(e)}"
        )