# Helpdesk con IA (LangGraph + RAG)

Sistema de soporte técnico que responde consultas automáticamente usando una base de conocimiento, y **escala a un agente humano** cuando la IA no tiene suficiente información.

## ¿Cómo funciona?

1. El usuario crea un ticket con su consulta.
2. **RAG:** se buscan los fragmentos relevantes en la documentación (`docs/`: manual de usuario, FAQ y guía de resolución de problemas) indexada en ChromaDB.
3. **Clasificación:** un LLM analiza la consulta, el contexto encontrado y la confianza de la búsqueda, y decide:
   - **Automático** → genera la respuesta con base en la documentación.
   - **Escalado** → el ticket queda pendiente para que un agente humano responda desde la interfaz.
4. El flujo completo está modelado como un grafo en **LangGraph**, con estado e historial por ticket.

## Tecnologías

Python · LangChain · LangGraph · OpenAI API (GPT-4o-mini, embeddings) · ChromaDB · SQLite · Streamlit

## Instalación

```bash
pip install streamlit langchain-openai langchain-community langchain-classic langchain-text-splitters langgraph langgraph-checkpoint-sqlite chromadb
```

Configura tu API key de OpenAI como variable de entorno:

```bash
# PowerShell
$env:OPENAI_API_KEY="tu-api-key"
# Linux / Mac
export OPENAI_API_KEY="tu-api-key"
```

## Uso

Desde la carpeta del proyecto:

```bash
streamlit run app.py
```

La primera vez, pulsa **"Configurar RAG"** para indexar los documentos de `docs/`.
