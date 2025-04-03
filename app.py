# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import openai
import uuid
import tempfile
from pathlib import Path
import time
import PyPDF2
import docx

# Carregar variáveis de ambiente
load_dotenv()

# Configurações do PostgreSQL
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

# Configurações para clientes OpenAI separados
# Cliente para Embeddings
embedding_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
)

# Cliente para Chat
chat_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
)

# Modelos específicos da Azure OpenAI
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Função para conexão com o banco de dados
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# Função para inicializar o banco de dados
def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Criar extensão pgvector se não existir
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Criar tabela para armazenar documentos e seus embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename TEXT,
                    chunk_number INTEGER,
                    content TEXT,
                    embedding VECTOR(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Criar índice para busca por similaridade
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Erro ao inicializar o banco de dados: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    return False

# Função para ler o conteúdo de arquivos
def extract_text_from_file(file):
    file_extension = Path(file.name).suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    text = ""
    try:
        if file_extension == ".pdf":
            with open(tmp_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        elif file_extension == ".docx":
            doc = docx.Document(tmp_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif file_extension == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        else:
            text = "Formato de arquivo não suportado."
    
    except Exception as e:
        text = f"Erro ao processar o arquivo: {e}"
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return text

# Função para dividir texto em chunks
def chunk_text(text, chunk_size=1000, overlap=100):
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

# Função para gerar embeddings usando Azure OpenAI
def generate_embedding(text):
    try:
        # Usar o cliente específico para embeddings
        response = embedding_client.embeddings.create(
            input=text,
            model=EMBEDDING_DEPLOYMENT
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

# Função para armazenar documentos e embeddings no banco de dados
def store_document(filename, chunks):
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            doc_id = uuid.uuid4()
            embedding = generate_embedding(chunk)
            
            if embedding:
                # Convert embedding to string format for PostgreSQL
                embedding_str = f"{embedding}"
                
                cursor.execute(
                    """
                    INSERT INTO documents (id, filename, chunk_number, content, embedding) 
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (str(doc_id), filename, i, chunk, embedding_str)
                )
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erro ao armazenar documento: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

        
# Função para buscar documentos similares
def search_similar_documents(query, limit=5):
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        query_embedding = generate_embedding(query)
        if not query_embedding:
            return []
        
        cursor = conn.cursor()
        
        # Convert Python list to a string representation that PostgreSQL can parse as a vector
        embedding_str = f"{query_embedding}"
        
        cursor.execute(
            """
            SELECT id, filename, chunk_number, content, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT %s
            """,
            (embedding_str, limit)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "filename": row[1],
                "chunk": row[2],
                "content": row[3],
                "similarity": row[4]
            })
        
        return results
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        return []
    finally:
        conn.close()


# Função para gerar resposta da Azure OpenAI com base nos documentos encontrados
def generate_response(query, search_results):
    if not search_results:
        return "Não encontrei informações relevantes para responder à sua pergunta."
    
    # Prepara o contexto com os documentos encontrados
    context = "\n\n".join([f"Documento: {res['filename']} (Chunk {res['chunk']})\n{res['content']}" for res in search_results])
    
    try:
        # Usar o cliente específico para chat
        response = chat_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Você é um assistente útil que responde perguntas com base nos documentos fornecidos. Use apenas as informações presentes nos documentos para responder."},
                {"role": "user", "content": f"Com base nos seguintes documentos:\n\n{context}\n\nResponda à pergunta: {query}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
        return f"Erro ao gerar resposta: {e}"

# Interface Streamlit
def main():
    st.set_page_config(page_title="Vetorização e Busca de Documentos", layout="wide")
    
    st.title("Vetorização e Busca de Documentos com Azure OpenAI")
    
    # Inicializar banco de dados
    db_status = init_db()
    if not db_status:
        st.error("Falha ao inicializar o banco de dados. Verifique se o PostgreSQL está rodando e as credenciais estão corretas.")
        return
    
    tab1, tab2 = st.tabs(["Upload de Documentos", "Busca Semântica"])
    
    # Tab para upload de documentos
    with tab1:
        st.header("Upload e Vetorização de Documentos")
        
        uploaded_file = st.file_uploader("Escolha um arquivo", type=["pdf", "txt", "docx"])
        
        if uploaded_file is not None:
            with st.spinner("Processando arquivo..."):
                # Extrair texto do arquivo
                text = extract_text_from_file(uploaded_file)
                
                if len(text) > 0:
                    st.success(f"Arquivo carregado com sucesso: {uploaded_file.name}")
                    
                    # Mostrar prévia do texto
                    with st.expander("Visualizar prévia do conteúdo"):
                        st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
                    
                    # Chunking e vetorização
                    if st.button("Processar e Vetorizar"):
                        chunks = chunk_text(text)
                        st.info(f"Texto dividido em {len(chunks)} chunks.")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, _ in enumerate(chunks):
                            status_text.text(f"Processando chunk {i+1}/{len(chunks)}...")
                            progress_bar.progress((i + 1) / len(chunks))
                            time.sleep(0.1)  # Simulação de progresso
                            
                        # Armazenar no banco de dados
                        if store_document(uploaded_file.name, chunks):
                            status_text.text("Processamento concluído!")
                            st.success(f"Documento '{uploaded_file.name}' vetorizado e armazenado com sucesso!")
                        else:
                            st.error("Erro ao armazenar o documento.")
                else:
                    st.error("Não foi possível extrair texto do arquivo.")
    
    # Tab para busca semântica
    with tab2:
        st.header("Busca Semântica nos Documentos")
        
        query = st.text_input("Digite sua pergunta:", "")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            search_button = st.button("Buscar")
            num_results = st.slider("Número de resultados", min_value=1, max_value=10, value=3)
        
        if search_button and query:
            with st.spinner("Buscando documentos relevantes..."):
                search_results = search_similar_documents(query, limit=num_results)
                
                if search_results:
                    st.subheader("Documentos Encontrados")
                    for i, res in enumerate(search_results):
                        with st.expander(f"{i+1}. {res['filename']} (Similaridade: {res['similarity']:.2f})"):
                            st.text(res['content'])
                    
                    st.subheader("Resposta Gerada")
                    with st.spinner("Gerando resposta..."):
                        response = generate_response(query, search_results)
                        st.write(response)
                else:
                    st.warning("Nenhum documento relevante encontrado.")

if __name__ == "__main__":
    main()