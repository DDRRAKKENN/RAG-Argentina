# 🇦🇷 RAG Argentina - Sistema de Preguntas y Respuestas Inteligente

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Una aplicación web interactiva que utiliza **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre Argentina. El sistema combina búsqueda semántica con modelos de lenguaje para proporcionar respuestas precisas y contextualizadas.

![App Preview](captura1.png)

## 🎯 Características

- ✅ **RAG (Retrieval-Augmented Generation)**: Combina búsqueda de documentos con generación de lenguaje
- 🤖 **Múltiples modelos LLM**: Soporte para Mistral, Mixtral y Flan-T5
- 🔍 **Búsqueda semántica**: Utiliza embeddings de Hugging Face y FAISS
- 💬 **Interfaz de chat**: Conversación natural con historial
- 📄 **Transparencia**: Muestra los documentos recuperados para cada respuesta
- ⚡ **Cache inteligente**: Optimización de modelos con Streamlit cache
- 🎨 **Diseño moderno**: UI intuitiva y responsive

## 📋 Tabla de Contenidos

- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Configuración](#️-configuración)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Cómo Funciona](#-cómo-funciona-rag)
- [Despliegue](#-despliegue)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                  │
│                            ↓                                     │
│                      Pregunta/Query                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Interfaz de chat                                       │  │
│  │  • Configuración de modelos                               │  │
│  │  • Visualización de resultados                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PROCESAMIENTO                         │
│                                                                  │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │  RETRIEVAL SYSTEM   │       │   GENERATION SYSTEM  │         │
│  │  (Búsqueda)         │       │   (Generación)       │         │
│  │                     │       │                      │         │
│  │  1. Embedding Model │       │  1. LLM Selection    │         │
│  │     ↓               │       │     ↓                │         │
│  │  2. Vector Query    │───────┤  2. Prompt Building  │         │
│  │     ↓               │       │     ↓                │         │
│  │  3. FAISS Search    │       │  3. Response Gen.    │         │
│  │     ↓               │       │                      │         │
│  │  4. Top-K Docs      │       │                      │         │
│  └─────────────────────┘       └─────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                                 │
│                                                                  │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │   VECTORSTORE       │       │   HUGGING FACE API   │         │
│  │   (FAISS)           │       │                      │         │
│  │                     │       │  • Mixtral-8x7B      │         │
│  │  • 20 documentos    │       │  • Mistral-7B        │         │
│  │  • 384 dimensiones  │       │  • Flan-T5           │         │
│  │  • Índice L2        │       │                      │         │
│  └─────────────────────┘       └─────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de Datos Detallado

```
1. INGESTA DE DOCUMENTOS
   ┌──────────────┐
   │  Documentos  │
   │  (texto)     │
   └──────┬───────┘
          │
          ↓
   ┌──────────────────┐
   │ Chunking/Split   │ (si es necesario)
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ Embedding Model  │ (HuggingFace)
   │ all-MiniLM-L6-v2 │
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │  Vectores 384D   │
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │  FAISS Index     │ (almacenamiento)
   └──────────────────┘

2. CONSULTA (QUERY TIME)
   ┌──────────────┐
   │ User Query   │
   └──────┬───────┘
          │
          ↓
   ┌──────────────────┐
   │ Embed Query      │ (mismo modelo)
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ Similarity Search│ (FAISS)
   │ Top-K retrieval  │
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ Documentos       │
   │ Relevantes       │
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ Prompt           │
   │ Construction     │
   │ (System +        │
   │  Context +       │
   │  Query)          │
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ LLM Generation   │ (HuggingFace API)
   └──────┬───────────┘
          │
          ↓
   ┌──────────────────┐
   │ Respuesta Final  │
   └──────────────────┘
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Token de Hugging Face ([Obtener aquí](https://huggingface.co/settings/tokens))

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/rag-argentina.git
cd rag-argentina
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
langchain-community>=0.0.1
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
huggingface-hub>=0.19.0
```

### Paso 4: Configurar variables de entorno (opcional)

```bash
# Crear archivo .env
echo "HUGGINGFACEHUB_API_TOKEN=tu_token_aqui" > .env
```

## 📖 Uso

### Ejecución local

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Primeros pasos

1. **Ingresar token**: En la barra lateral, ingresa tu token de Hugging Face
2. **Configurar modelos**: Selecciona el modelo de embeddings y LLM
3. **Ajustar parámetros**: Define el número de documentos a recuperar (k)
4. **Inicializar**: Presiona "🚀 Inicializar Sistema RAG"
5. **Chatear**: ¡Empieza a hacer preguntas sobre Argentina!

### Ejemplos de preguntas

```
- ¿Cuál es la capital de Argentina?
- ¿Qué montaña importante hay en Argentina?
- Háblame sobre el tango
- ¿Cuándo ganó Argentina la Copa Mundial?
- ¿Qué es el mate?
- ¿Cuáles son los límites de Argentina?
```

## ⚙️ Configuración

### Modelos disponibles

#### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2` (recomendado - rápido)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilenguaje)

#### LLMs
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (más potente)
- `mistralai/Mistral-7B-Instruct-v0.2` (balanceado)
- `google/flan-t5-large` (más rápido)

### Parámetros ajustables

- **k (Top-K)**: Número de documentos a recuperar (1-5)
- **max_tokens**: Longitud máxima de respuesta (en el código)
- **temperature**: Creatividad de las respuestas (en el código)

## 📁 Estructura del Proyecto

```
rag-argentina/
│
├── app.py                      # Aplicación principal de Streamlit
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
│
├── data/                      # (Opcional) Datos adicionales
│   └── documents.json         # Documentos sobre Argentina
|
└── notebooks/                 # Notebooks de desarrollo y explicaciones
    └── explicacion_RAG.ipynb
```

## 🔍 Cómo Funciona RAG

### ¿Qué es RAG?

**Retrieval-Augmented Generation (RAG)** es una técnica que combina:

1. **Retrieval (Recuperación)**: Búsqueda de información relevante en una base de datos
2. **Generation (Generación)**: Uso de un LLM para generar respuestas basadas en la información recuperada

### Proceso paso a paso

```python
# 1. PREPARACIÓN (Una vez)
documentos = ["doc1", "doc2", ...]
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documentos, embeddings)

# 2. CONSULTA (Cada pregunta)
query = "¿Cuál es la capital de Argentina?"

# 3. BÚSQUEDA
docs_relevantes = vectorstore.similarity_search(query, k=3)

# 4. CONSTRUCCIÓN DEL PROMPT
prompt = f"""
Contexto: {docs_relevantes}
Pregunta: {query}
Responde basándote SOLO en el contexto.
"""

# 5. GENERACIÓN
respuesta = llm.invoke(prompt)
```

### Ventajas de RAG

✅ **Información actualizada**: No requiere reentrenar el modelo  
✅ **Transparencia**: Puedes ver qué documentos se usaron  
✅ **Precisión**: Respuestas basadas en datos específicos  
✅ **Eficiencia**: No necesitas un modelo enorme  
✅ **Control**: Puedes agregar/quitar documentos fácilmente  

## 🌐 Despliegue

### Streamlit Cloud (Gratis)

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Configura los secrets (token de HF)
5. ¡Despliega!

**Configurar secrets:**
```toml
# .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN = "tu_token_aqui"
```

### Hugging Face Spaces

1. Crea un Space en [huggingface.co/spaces](https://huggingface.co/spaces)
2. Selecciona "Streamlit" como SDK
3. Sube tu código
4. Configura el token en los Settings

### Docker (Avanzado)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t rag-argentina .
docker run -p 8501:8501 rag-argentina
```

## 🛠️ Personalización

### Agregar tus propios documentos

```python
# En app.py, modifica la lista DOCUMENTS
DOCUMENTS = [
    "Tu documento 1",
    "Tu documento 2",
    # ... más documentos
]
```

### Cambiar el prompt del sistema

```python
# En la función generate_response
SystemMessage(content=f"""
Eres un experto en [TU TEMA].
Responde de manera [TU ESTILO].
...
""")
```

### Agregar soporte para archivos

```python
uploaded_file = st.file_uploader("Sube un PDF", type=['pdf'])
if uploaded_file:
    # Procesar archivo
    # Agregar al vectorstore
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Áreas de mejora

- [ ] Soporte para más formatos de archivo (PDF, DOCX, CSV)
- [ ] Integración con bases de datos vectoriales cloud
- [ ] Sistema de evaluación de respuestas
- [ ] Métricas y analytics
- [ ] Tests unitarios
- [ ] Soporte multilenguaje

## 📊 Tecnologías Utilizadas

### 🎨 **[Streamlit](https://streamlit.io/)** - Framework Web
Framework de Python que convierte scripts en aplicaciones web interactivas en minutos. Sin necesidad de HTML, CSS o JavaScript.

**¿Por qué Streamlit?**
- ✅ Desarrollo rápido (menos código)
- ✅ Perfecto para prototipos de ML/AI
- ✅ Componentes interactivos listos para usar
- ✅ Despliegue gratis en Streamlit Cloud

### 🔗 **[LangChain](https://langchain.com/)** - Orquestación de LLMs
Framework que facilita la construcción de aplicaciones con modelos de lenguaje. Piensa en él como un "pegamento" que conecta diferentes piezas de IA.

**¿Qué hace LangChain?**
- 🔌 Conecta LLMs con fuentes de datos
- 📝 Maneja prompts y plantillas
- 🔄 Crea cadenas de procesamiento (chains)
- 💾 Gestiona memoria y contexto

### 🤗 **[Hugging Face](https://huggingface.co/)** - Hub de Modelos de IA
Plataforma que aloja miles de modelos de IA pre-entrenados. Como "GitHub pero para modelos de inteligencia artificial".

**En este proyecto usamos:**
- 🤖 **LLMs**: Modelos que generan texto (Mistral, Mixtral)
- 🧠 **Embeddings**: Modelos que convierten texto en números
- 🔑 **API**: Para usar modelos sin descargarlos

### 🔍 **[FAISS](https://github.com/facebookresearch/faiss)** - Búsqueda Vectorial
Biblioteca de Facebook para búsqueda rápida de similitud entre vectores. Es como un "Google" pero para encontrar textos parecidos.

**¿Cómo funciona?**
```
Texto → Números (vector) → Índice FAISS → Búsqueda ultrarrápida
```

**Ventajas:**
- ⚡ Búsquedas en milisegundos (incluso con millones de documentos)
- 💾 Eficiente en memoria
- 📊 Diferentes algoritmos de búsqueda

### 🎯 **[Sentence Transformers](https://www.sbert.net/)** - Embeddings Semánticos
Modelos especializados en convertir frases/documentos en vectores que capturan su significado.

**Ejemplo simple:**
```python
"perro" → [0.2, 0.8, 0.1, ...]  # Vector
"gato"  → [0.3, 0.7, 0.2, ...]  # Vector similar

# Distancia corta = significados parecidos
```

**¿Por qué son importantes?**
- 🔤 Entienden significado, no solo palabras exactas
- 🌍 Funcionan en múltiples idiomas
- 🎯 Base del sistema de búsqueda semántica

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Tu Nombre**
- GitHub: [Jorge Ignacio Lara Ceballos](https://github.com/DDRRAKKENN)
- LinkedIn: [Jorge Ignacio Lara Ceballos](https://www.linkedin.com/in/jorge-ignacio-lara-ceballos/)

## 🙏 Agradecimientos

- Anthropic por Claude
- Hugging Face por los modelos
- Comunidad de LangChain
- Comunidad de Streamlit

## 📚 Referencias y Recursos

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [FAISS Documentation](https://faiss.ai/)

---

⭐ Si te gustó este proyecto, ¡dale una estrella en GitHub!

💬 ¿Preguntas? Abre un [issue](https://github.com/tu-usuario/rag-argentina/issues)
