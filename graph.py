from typing import TypedDict, Annotated, List, Optional, Dict, Any
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from rag_system import VectorRAGSystem
from config import *

class HelpdeskState(TypedDict):
    consulta: str
    categoria: str
    respuesta_rag: Optional[str]
    confianza: float
    fuentes: List[str]
    contexto_rag: Optional[str]
    requiere_humano: bool
    respuesta_humano: Optional[str]
    respuesta_final: Optional[str]
    historial: Annotated[List[str], add]

class HelpdeskGraph:
    #grafo del sistema helpdesk

    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL, temperature=0.1)
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)
        self.graph=None

    def procesar_rag(self, state):
        #bsuca el contexto de la consulta utilizando el sistema rag
        consulta = state['consulta']
        resultado = self.rag.buscar(consulta)
        return {
            "respuesta_rag": resultado['respuesta'],
            "confianza": resultado['confianza'],
            "fuentes": resultado['fuentes'],
            "contexto_rag": resultado['respuesta'],
            "historial": [
                f"Rag ejecutado con multiqueryretriever",
                f"Confianza: {resultado["confianza"]}",
                f"Fuentes: consultadas: {len(resultado['fuentes'])}"
            ] 
        }

    def clasificar_con_contexto(self, state):
        #clasifica la consulta para responder automaticamente o escalar con el contesto del rag
        consulta = state['consulta']
        contexto_rag = state.get('contexto_rag', '')
        confianza= state.get('confianza', 0)

        prompt = ChatPromptTemplate.from_template(
            """Analiza esta consulta de helpdesk y decide si puede responderse automáticamente o necesita escalado:

            CONSULTA DEL USUARIO: {consulta}

            INFORMACIÓN ENCONTRADA EN LA BASE DE CONOCIMIENTO:
            {contexto_rag}

            CONFIANZA DE LA BÚSQUEDA: {confianza}

            Criterios de decisión:
            - AUTOMATICO: Si la información de la BD responde completamente la consulta, 
            tiene buena confianza (>0.6), y es un tema estándar/procedimiento conocido
  
            - ESCALADO: Si la información es insuficiente, confianza baja, problema complejo/único,
            requiere acceso a sistemas internos, o involucra decisiones de negocio

            Responde solo con "automatico" o "escalado" y una breve justificación (máximo 20 palabras):"""
        )

        try:
            response = self.llm.invoke(prompt.format(
                consulta=consulta,
                contexto_rag=contexto_rag,
                confianza=confianza
            ))

            content = response.content.strip().lower()

            if "automatico" in content or "automático" in content:
                contegoria = "automatico"
            elif "escalado" in content:
                categoria= "escalado"
            else:
                categoria= "automatico" if confianza >= 0.6 else "escalado"

            return{
                "categoria": categoria,
                "historial": [
                    f"Clasificacion con contexto: {categoria}",
                    f"Justificacion: {response.content}"
                ]
            }

        except Exception as e:
            categoria = "automatico" if confianza >= 0.60 else "escalado"
            return{
                "categoria": categoria,
                "historial": [
                    f"Error en la clasificacion, usando confianza: {confianza}"
                ]
            }

    def preparar_escalado(self, state):
        #preparar el escalado a un humano
        return {
            "requiere_humano": True,
            "historial": ["Ecalado a agente humano - esperando intervencion."]
        }

    def procesar_respuesta_humano(self, state):
        #procesa la respuesta del humano
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return{
                "respuesta_final": respuesta_humano,
                "historial": ["agente humano proporciono una respuesta."]
            }

        return {
            "historial": ["Esperando respuesta del agente humano."]
        }

    def generar_respuesta_final(self, state):
        #genera la respuesta final
        if state.get("respuesta_final"):
            return{
                "historial": ["respuesta final proporcionada pro agente humano"]
            }

        respuesta_rag = state.get("respuesta_rag", "")
        fuentes = state.get("fuentes", [])

        respuesta_final = respuesta_rag
        if fuentes:
            fuentes_textos = ", ".join(fuentes)
            respuesta_final += f"\n\nFuentes consultas: {fuentes_textos}"

        return {
            "respuesta_final": respuesta_final,
            "historial": ["respuesta final generada automaticamente."]
        }

    #funciones de enrutamiento

    def decidir_desde_clasificacion(self,state):
        #decide hacia donde ir despues de la clasificacicon con contexto rag
        categoria = state.get("categoria", "escalado")
        if categoria == "automatico":
            return "respuesta_final"
        else:
            return "escalado"

    def decidir_desde_humano(self, state):
        #decide si continuar o esperar respuesta humana
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return "procesar_humano"
        else:
            return "esperar"

    def crear_grafo(self):
        #crear el grafo de langraph y control de flujo
        graph = StateGraph(HelpdeskState)

        graph.add_node("rag", self.procesar_rag)
        graph.add_node("clasificar", self.clasificar_con_contexto)
        graph.add_node("escalado", self.preparar_escalado)
        graph.add_node("respuesta_final", self.generar_respuesta_final)
        graph.add_node("procesar_humano", self.procesar_respuesta_humano)

        #definir estructura del grafo
        graph.add_edge(START, "rag")
        graph.add_edge("rag", "clasificar")

        graph.add_conditional_edges(
            "clasificar", self.decidir_desde_clasificacion,
            {
                "respuesta_final": "respuesta_final",
                "escalado": "escalado"
            }
        )

        graph.add_conditional_edges(
            "escalado", self.decidir_desde_humano,
            {
                "procesar_humano": "procesar_humano",
                "esperar": END
            }
        )

        graph.add_edge("procesar_humano", END)
        graph.add_edge("respuesta_final", END)

        self.graph = graph

        return graph

    def compilar(self):
        #compila el grafo checkpointer
        if not self.graph:
            self.crear_grafo()

        conn = sqlite3.connect("helpdesk.db", check_same_thread=False)

        checkpointer = SqliteSaver(conn)

        compiled = self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["procesar_humano"]
        )

        return compiled

def crear_helpdesk():
    helpdesk = HelpdeskGraph()
    return helpdesk.compilar()        