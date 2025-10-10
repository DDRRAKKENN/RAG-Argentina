import streamlit as st
import os
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

# Configuración de la página
st.set_page_config(
    page_title="RAG Argentina 🇦🇷",
    page_icon="🤖",
    layout="wide"
)

# Dataset de Argentina
DOCUMENTS = [
    "Argentina es el segundo país más grande de América del Sur y el octavo más grande del mundo.",
    "Buenos Aires es la capital y ciudad más grande de Argentina.",
    "El tango es un baile y género musical originario de Argentina, especialmente de Buenos Aires.",
    "Argentina limita con Chile, Bolivia, Paraguay, Brasil y Uruguay.",
    "El Aconcagua, ubicado en Argentina, es la montaña más alta de América con 6,961 metros.",
    "Argentina es famosa por su carne de res de alta calidad y sus asados tradicionales.",
    "El idioma oficial de Argentina es el español, con un acento distintivo rioplatense.",
    "Las Cataratas del Iguazú, compartidas con Brasil, son una de las maravillas naturales del mundo.",
    "Argentina ganó la Copa Mundial de la FIFA en 1978, 1986 y 2022.",
    "El peso argentino es la moneda oficial del país.",
    "Diego Maradona y Lionel Messi son dos de los futbolistas más famosos de Argentina.",
    "La Pampa argentina es una extensa llanura conocida por su producción agrícola y ganadera.",
    "El glaciar Perito Moreno es uno de los glaciares más famosos y accesibles del mundo.",
    "Argentina declaró su independencia de España el 9 de julio de 1816.",
    "El mate es la bebida tradicional argentina, consumida en todo el país.",
    "La Patagonia argentina es conocida por sus paisajes impresionantes y vida silvestre única.",
    "Argentina es uno de los principales productores y exportadores de vino del mundo.",
    "La región vinícola de Mendoza es famosa por su producción de vino Malbec.",
    "El Teatro Colón en Buenos Aires es uno de los teatros de ópera más importantes del mundo.",
    "Argentina tiene una población de aproximadamente 45 millones de habitantes.",
]

# Título
st.title("🇦🇷 Chat RAG sobre Argentina")
st.markdown("### Pregúntame sobre Argentina usando RAG con Hugging Face")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Token de Hugging Face
    hf_token = st.text_input(
        "Token de Hugging Face",
        type="password",
        help="Obtén tu token en: https://huggingface.co/settings/tokens"
    )
    
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    # Modelo de embeddings
    st.subheader("Modelo de Embeddings")
    embedding_model_name = st.selectbox(
        "Selecciona el modelo",
        ["sentence-transformers/all-MiniLM-L6-v2", 
         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        help="Modelo para crear embeddings de los documentos"
    )
    
    # Modelo LLM
    st.subheader("Modelo LLM")
    llm_model = st.selectbox(
        "Selecciona el LLM",
        ["mistralai/Mixtral-8x7B-Instruct-v0.1",
         "mistralai/Mistral-7B-Instruct-v0.2",
         "google/flan-t5-large"]
    )
    
    # Número de documentos a recuperar
    k_docs = st.slider("Documentos a recuperar (k)", 1, 5, 3)
    
    # Botón para inicializar
    initialize = st.button("🚀 Inicializar Sistema RAG", type="primary")
    
    st.divider()
    st.markdown("**Información del sistema:**")
    if "vectorstore" in st.session_state:
        st.success(f"✅ Vectorstore inicializado")
        st.info(f"📊 {len(DOCUMENTS)} documentos indexados")
    else:
        st.warning("⚠️ Sistema no inicializado")

# Función para inicializar el vectorstore
@st.cache_resource
def init_vectorstore(embedding_model_name):
    """Inicializa el vectorstore con los documentos"""
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    docs = [Document(page_content=text) for text in DOCUMENTS]
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

# Función para inicializar el LLM
@st.cache_resource
def init_llm(model_name):
    """Inicializa el modelo de lenguaje"""
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation"
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

# Función para generar respuesta
def generate_response(query, context, chat_model):
    """Genera respuesta usando RAG"""
    messages = [
        SystemMessage(content=f"""Eres un guía turístico Argentino muy gracioso y hablas como argentino.
                      Solo responde teniendo en cuenta el contexto proporcionado.
                      Si no sabes la respuesta di 'No lo sé, esa información no está en mis documentos'
                      """),
        HumanMessage(content=f"Responde a la pregunta: '{query}' basándote en estos documentos: {context}")
    ]
    
    ai_response = chat_model.invoke(messages)
    return ai_response.content

# Inicialización del sistema
if initialize:
    if not hf_token:
        st.error("❌ Por favor, ingresa tu token de Hugging Face")
    else:
        with st.spinner("🔄 Inicializando sistema RAG..."):
            try:
                # Inicializar vectorstore
                st.session_state.vectorstore = init_vectorstore(embedding_model_name)
                st.session_state.chat_model = init_llm(llm_model)
                st.session_state.messages = []
                st.success("✅ Sistema inicializado correctamente!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error al inicializar: {str(e)}")

# Interfaz de chat
if "vectorstore" in st.session_state and "chat_model" in st.session_state:
    
    # Mostrar historial de chat
    st.subheader("💬 Conversación")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "docs" in message:
                with st.expander("📄 Ver documentos recuperados"):
                    for i, doc in enumerate(message["docs"], 1):
                        st.caption(f"{i}. {doc.page_content}")
    
    # Input del usuario
    if prompt := st.chat_input("Hazme una pregunta sobre Argentina..."):
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Buscando información..."):
                try:
                    # Recuperar documentos relevantes
                    retrieved_docs = st.session_state.vectorstore.similarity_search(
                        prompt, 
                        k=k_docs
                    )
                    
                    # Generar respuesta
                    response = generate_response(
                        prompt, 
                        retrieved_docs, 
                        st.session_state.chat_model
                    )
                    
                    st.markdown(response)
                    
                    # Mostrar documentos recuperados
                    with st.expander("📄 Ver documentos recuperados"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.caption(f"{i}. {doc.page_content}")
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "docs": retrieved_docs
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error al generar respuesta: {str(e)}")
    
    # Botón para limpiar conversación
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("👈 Configura el sistema en la barra lateral y presiona 'Inicializar Sistema RAG'")
    
    # Ejemplos de preguntas
    st.subheader("💡 Ejemplos de preguntas:")
    st.markdown("""
    - ¿Cuál es la capital de Argentina?
    - ¿Qué montaña importante hay en Argentina?
    - Háblame sobre el tango
    - ¿Cuándo ganó Argentina la Copa Mundial?
    - ¿Qué es el mate?
    - ¿Qué bebida tradicional hay en Argentina?
    """)

# Footer
st.divider()
st.caption("🤖 Sistema RAG con Hugging Face y LangChain | Desarrollado con Streamlit")