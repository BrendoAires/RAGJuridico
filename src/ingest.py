import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Carrega a chave contida no arquivo .env
load_dotenv()

def processar_processo_juridico():
    print("=== Iniciando Ingestão do Processo Jurídico ===")
    
    # 1. Verificar se existem documentos na pasta data/
    if not os.path.exists("data") or not os.listdir("data"):
        print("Erro: A pasta 'data' está vazia ou não existe. Adicione os PDFs do processo nela.")
        return

    # 2. Carregar os documentos
    print("Lendo PDFs da pasta 'data'...")
    loader = PyPDFDirectoryLoader("data")
    documentos = loader.load()
    print(f"Total de páginas lidas: {len(documentos)}")

    # 3. Dividir o texto em blocos menores (chunks) mantendo o contexto das frases
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        length_function=len
    )
    blocos = text_splitter.split_documents(documentos)
    print(f"Total de blocos de texto gerados: {len(blocos)}")

    # 4. Inicializar os Embeddings do Google (Gratuito/Barato)
    # Usamos o modelo padrão de texto do Google para embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 5. Criar o Banco Vetorial Local (Chroma)
    db_diretorio = "chroma_db"
    print("Processando textos e criando banco vetorial...")
    
    vector_db = Chroma.from_documents(
        documents=blocos,
        embedding=embeddings,
        persist_directory=db_diretorio
    )
    
    print(f"Sucesso! Banco vetorial salvo na pasta '{db_diretorio}'.")

if __name__ == "__main__":
    processar_processo_juridico()