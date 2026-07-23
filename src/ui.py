import gradio as gr
import requests
import os

# URL da nossa API FastAPI
# No Docker Compose, o hostname do backend é 'backend'. Fora do Docker, é '127.0.0.1'
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/consultar")

def consultar_api(pergunta: str):
    if not pergunta.strip():
        return "Por favor, digite uma pergunta válida.", ""

    payload = {"pergunta": pergunta}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            dados = response.json()
            resposta_texto = dados.get("resposta", "Sem resposta gerada.")
            fontes = dados.get("fontes", [])
            
            # Formatação estilizada das fontes consultadas
            if not fontes:
                fontes_formatadas = "### 📄 Documentos Citados / Trechos:\n\nNenhuma fonte específica citada."
            else:
                fontes_formatadas = "### 📄 Documentos Citados / Trechos:\n\n"
                for idx, f in enumerate(fontes, 1):
                    # Garante suporte caso a fonte venha como dict ou string
                    if isinstance(f, dict):
                        arquivo = f.get('arquivo', 'Desconhecido')
                        trecho = f.get('conteudo', '')
                    else:
                        arquivo = str(f)
                        trecho = ""
                    
                    fontes_formatadas += f"**[{idx}] Arquivo:** `{arquivo}`\n> {trecho}\n\n---\n"
                
            return resposta_texto, fontes_formatadas
        else:
            erro_detail = response.json().get("detail", "Erro desconhecido na API.")
            return f"❌ Erro na API ({response.status_code}): {erro_detail}", ""

    except requests.exceptions.ConnectionError:
        return "❌ Não foi possível conectar à API FastAPI. Certifique-se de que o backend está rodando na porta 8000.", ""
    except Exception as e:
        return f"❌ Erro inesperado: {str(e)}", ""

# Interface Gradio
with gr.Blocks(title="LegalMind Analytics - Assistente RAG Jurídico", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # ⚖️ LegalMind Analytics — Assistente Processual RAG
        ### Análise e extração de informações de processos jurídicos com inteligência artificial auditável.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_pergunta = gr.Textbox(
                lines=3, 
                placeholder="Ex: Quais são os principais argumentos apresentados pela defesa?", 
                label="Digite sua dúvida sobre o processo:"
            )
            btn_enviar = gr.Button("🔍 Consultar Processo", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=2):
            output_resposta = gr.Markdown(label="Resposta do Assistente")
        with gr.Column(scale=1):
            output_fontes = gr.Markdown(label="Fontes de Auditoria")

    btn_enviar.click(
        fn=consultar_api,
        inputs=[input_pergunta],
        outputs=[output_resposta, output_fontes]
    )

# PONTO CHAVE: Adicionado server_name="0.0.0.0" para o Docker expor o painel correto!
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)