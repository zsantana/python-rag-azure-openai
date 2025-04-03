# Vetorização e Busca Semântica de Documentos com Azure OpenAI

## Descrição do Projeto

Este projeto é uma aplicação Streamlit que permite a vetorização e busca semântica de documentos utilizando Azure OpenAI, PostgreSQL com extensão pgvector e técnicas de Retrieval-Augmented Generation (RAG).

## Funcionalidades Principais

- Upload de documentos (PDF, DOCX, TXT)
- Processamento de texto com chunking
- Geração de embeddings usando Azure OpenAI
- Armazenamento de documentos vetorizados no PostgreSQL
- Busca semântica com recuperação de documentos similares
- Geração de respostas contextuais usando modelo de chat da Azure OpenAI

## Pré-requisitos

- Python 3.12
- PostgreSQL com extensão pgvector
- Conta Azure OpenAI
- Bibliotecas Python:
  - streamlit
  - psycopg2
  - openai
  - python-dotenv
  - PyPDF2
  - python-docx
  - numpy
  - pandas

## Configuração de Ambiente

1. Clone o repositório
2. Crie um ambiente virtual
3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Crie um arquivo `.env` com as seguintes variáveis:
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

AZURE_OPENAI_EMBEDDING_API_KEY=sua_chave_embedding
AZURE_OPENAI_EMBEDDING_API_VERSION=versão_api
AZURE_OPENAI_EMBEDDING_ENDPOINT=endpoint_embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=nome_deployment_embedding

AZURE_OPENAI_CHAT_API_KEY=sua_chave_chat
AZURE_OPENAI_CHAT_API_VERSION=versão_api
AZURE_OPENAI_CHAT_ENDPOINT=endpoint_chat
AZURE_OPENAI_CHAT_DEPLOYMENT=nome_deployment_chat
```

## Executando a Aplicação

```bash
streamlit run app.py
```

## Arquitetura

- Frontend: Streamlit
- Vetorização: Azure OpenAI Embeddings
- Banco de Dados: PostgreSQL com pgvector
- Geração de Respostas: Azure OpenAI Chat

## Fluxo de Trabalho

1. Upload de Documento
2. Chunking de Texto
3. Geração de Embeddings
4. Armazenamento no Banco de Dados
5. Busca Semântica
6. Geração de Resposta Contextual

## Contribuição

Pull requests são bem-vindos. Para mudanças importantes, abra primeiro uma issue para discutir o que você gostaria de modificar.

## Licença

[MIT](https://choosealicense.com/licenses/mit/)
