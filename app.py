import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ---------- CONFIGURACIÓN GENERAL ----------
st.set_page_config(page_title="Análisis Inteligente de PDF", layout="centered", initial_sidebar_state="collapsed")

# ---------- ESTILO PERSONALIZADO ----------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif !important;
            color: #1e1e1e;
        }

        body {
            background: linear-gradient(135deg, #c9d6ff, #e2e2e2);
        }

        .stApp {
            background: linear-gradient(135deg, #c9d6ff, #e2e2e2);
        }

        .main {
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        h1 {
            color: #2e3a59;
            text-align: center;
            font-weight: 700;
        }

        h3, h2, h4 {
            color: #2e3a59;
        }

        .stTextInput > div > div > input, .stTextArea textarea {
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            border: 1px solid #a1a1a1;
            color: #1e1e1e;
        }

        .stFileUploader {
            background-color: rgba(255,255,255,0.7);
            border-radius: 12px;
            padding: 1rem;
            transition: all 0.3s ease-in-out;
        }

        .stFileUploader:hover {
            box-shadow: 0 0 12px rgba(46,58,89,0.2);
        }

        .stButton > button {
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6rem 1.2rem;
            transition: 0.3s;
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #764ba2, #667eea);
            transform: scale(1.03);
        }

        .stSidebar {
            background: rgba(255,255,255,0.7);
            border-radius: 15px;
        }

        .stAlert {
            border-radius: 10px;
        }

        .stMarkdown {
            color: #1e1e1e;
        }

        footer {visibility: hidden;}

        .centered-image {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- TÍTULO Y PRESENTACIÓN ----------
st.title('📄 Análisis Inteligente de PDF con RAG')
st.write("🧠 **Versión de Python:**", platform.python_version())

# ---------- IMAGEN CENTRADA ----------
try:
    image = Image.open('Chat_pdf.png')
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(image, width=320)
    st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.subheader("📘 Asistente de Análisis de PDF")
    st.markdown("""
    Este agente utiliza **Generación Aumentada por Recuperación (RAG)** para analizar el contenido del PDF cargado.
    Puedes hacerle preguntas específicas sobre el texto, y responderá con información relevante extraída del documento.
    """)

# ---------- CLAVE API ----------
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# ---------- SUBIDA DE PDF ----------
pdf = st.file_uploader("📤 Carga el archivo PDF que deseas analizar", type="pdf")

# ---------- PROCESAMIENTO DEL PDF ----------
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📝 Texto extraído: **{len(text)} caracteres**")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"📚 Documento dividido en **{len(chunks)} fragmentos**")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.subheader("💬 Realiza una pregunta sobre el documento")
        user_question = st.text_area(" ", placeholder="Ejemplo: ¿Cuál es el tema principal del documento?")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.markdown("### 🧾 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📎 Carga un archivo PDF para comenzar el análisis")

