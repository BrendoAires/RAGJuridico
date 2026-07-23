# Imagem base oficial do Python
FROM python:3.10-slim

# Evita que o Python grave arquivos .pyc no disco e habilita logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho no container
WORKDIR /app

# Instala dependências do sistema operacional necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte e arquivos da aplicação
COPY . .

# Expõe as portas do FastAPI (8000) e do Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Comando padrão (pode ser sobrescrito pelo docker-compose)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]