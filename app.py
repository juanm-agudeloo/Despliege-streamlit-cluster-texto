import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import spacy
from spacy.cli import download  # <-- Importamos la herramienta de descarga aquí arriba

st.set_page_config(page_title="Clustering de Texto Avanzado", page_icon="🧩", layout="wide")
st.title("🧩 Agrupador Automático de Textos")
st.markdown("Clustering semántico con limpieza avanzada (NLP) y nombrado dinámico de grupos.")

# --- CARGA DE MODELOS (Caché para rendimiento) ---
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_spacy_model():
    try:
        nlp_model = spacy.load("es_core_news_sm")
    except OSError:
        # Usamos la función directamente para no confundir el scope local
        download("es_core_news_sm")
        nlp_model = spacy.load("es_core_news_sm")
    return nlp_model

model = load_embeddings_model()
nlp = load_spacy_model()

# --- FUNCIONES DE LIMPIEZA Y NLP ---
def limpiar_texto_avanzado(texto):
    """Limpia el texto conservando ÚNICAMENTE sustantivos y nombres propios."""
    doc = nlp(texto)
    tokens_limpios = []
    
    for token in doc:
        if not token.is_punct and not token.is_space and len(token.text) >= 3:
            # Lista blanca: Solo Sustantivos (NOUN) y Nombres Propios (PROPN)
            if token.pos_ in ["NOUN", "PROPN"]:
                tokens_limpios.append(token.lemma_.lower())
                
    return " ".join(tokens_limpios)

# Ampliamos un poco las stop words por seguridad
spanish_stop_words = [
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "ante", 
    "con", "en", "para", "por", "y", "o", "u", "que", "es", "son", "se", "lo", "al", 
    "su", "sus", "te", "me", "mi", "muy", "más", "como", "pero", "este", "esta", 
    "esto", "ha", "han", "he", "ya", "no", "si", "ser", "estar", "tener", "hacer",
    "mucho", "mucha", "muchos", "muchas", "todo", "toda", "todos", "todas", "uno", "tanto"
]
def extraer_palabras_clave_expandido(df, stopwords):
    palabras_por_cluster = {}
    for c in df['ID Clúster'].unique():
        if c == -1:
            palabras_por_cluster[c] = ["Ruido", "Misceláneo"]
            continue
            
        # Usamos la columna del texto limpio (sin adjetivos, sin palabras cortas)
        textos_cluster = df[df['ID Clúster'] == c]['Texto Limpio'].tolist()
        
        if len(textos_cluster) < 2:
            palabras_por_cluster[c] = ["Específico", "Único"]
            continue
            
        # TF-IDF para sacar las palabras más representativas
        vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=8) # Subimos a 8 palabras
        try:
            X = vectorizer.fit_transform(textos_cluster)
            palabras = vectorizer.get_feature_names_out()
            importancias = X.sum(axis=0).A1
            # Ordenamos por peso de importancia
            palabras_ordenadas = [palabra for _, palabra in sorted(zip(importancias, palabras), reverse=True)]
            palabras_por_cluster[c] = palabras_ordenadas
        except Exception:
            palabras_por_cluster[c] = ["Indefinido", "Mixto"]
            
    return palabras_por_cluster

def generar_titulo_y_resumen(cluster_idx, palabras):
    if cluster_idx == -1:
        titulo = "Ruido (Atípicos)"
        resumen = "🌪️ **Clúster Ruido:** Contiene opiniones aisladas, inclasificables o que divergen demasiado de las tendencias principales del conjunto de datos."
        return titulo, resumen
    
    # Capitalizamos las palabras
    palabras = [p.capitalize() for p in palabras]
    
    # Algoritmo de Nombrado Expandido (Combina los dos temas principales)
    if len(palabras) >= 2:
        titulo = f"{palabras[0]} & {palabras[1]}"
        conceptos_extra = ", ".join(palabras[2:6]) if len(palabras) > 2 else "conceptos similares"
        
        resumen = (f"👥 **Clúster {cluster_idx} - {titulo}:** Este grupo está conformado por los textos que "
                   f"discuten directamente temas relacionados con **{palabras[0]}** y **{palabras[1]}**. "
                   f"Al analizar la estructura de estos mensajes, también se detecta una fuerte presencia de conceptos como: *{conceptos_extra}*.")
    else:
        titulo = palabras[0] if palabras else "General"
        resumen = f"👥 **Clúster {cluster_idx} - {titulo}:** Agrupa a las personas con comentarios altamente focalizados de forma exclusiva en el concepto de *{titulo}*."
    
    return titulo, resumen

# --- BARRA LATERAL (Configuración) ---
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    algoritmo = st.selectbox("Algoritmo de Clustering", ["K-Means", "DBSCAN"])
    if algoritmo == "K-Means":
        num_clusters = st.slider("Número de clústeres (k)", min_value=2, max_value=15, value=3)
    else:
        eps = st.slider("Distancia máxima (eps)", min_value=0.1, max_value=2.0, value=0.5)
        min_samples = st.slider("Mínimo de muestras", min_value=2, max_value=10, value=2)
        
    st.header("📊 Configuración de Gráfica")
    dim_grafica = st.radio("Dimensionalidad", ["2D", "3D"], horizontal=True)

# --- ÁREA PRINCIPAL (Entrada de datos) ---
st.subheader("1. Entrada de Datos")
opcion_entrada = st.radio("¿Fuente de los datos?", ("Escribir texto", "Subir archivo (.txt o .csv)"), horizontal=True)

textos_raw = []
if opcion_entrada == "Escribir texto":
    texto_usuario = st.text_area("Ingresa los textos a analizar (separa los documentos o párrafos por saltos de línea):", height=200)
    textos_raw = [t.strip() for t in texto_usuario.split('\n') if t.strip() != '']
else:
    archivo_subido = st.file_uploader("Selecciona un archivo", type=["txt", "csv"])
    if archivo_subido is not None:
        if archivo_subido.name.endswith('.csv'):
            df_input = pd.read_csv(archivo_subido)
            columna_texto = 'Comentarios' if 'Comentarios' in df_input.columns else df_input.columns[0]
            textos_raw = df_input[columna_texto].dropna().astype(str).tolist()
        else:
            texto_usuario = archivo_subido.getvalue().decode("utf-8")
            textos_raw = [t.strip() for t in texto_usuario.split('\n') if t.strip() != '']
        st.success(f"Archivo cargado correctamente. Se detectaron {len(textos_raw)} líneas para procesar.")

# --- EJECUCIÓN Y RESULTADOS ---
st.subheader("2. Resultados del Análisis")

if st.button("Ejecutar Clustering 🚀"):
    if len(textos_raw) == 0:
        st.warning("⚠️ Por favor, proporciona textos antes de ejecutar el modelo.")
    else:
        with st.spinner('Aplicando NLP: Limpiando adjetivos, filtrando longitud, vectorizando y agrupando...'):
            
            # Limpieza NLP (usada solo para extracción de etiquetas y temas)
            textos_limpios = [limpiar_texto_avanzado(t) for t in textos_raw]
            
            # Vectorización (mantenemos el texto raw para que el Transformer entienda el contexto)
            embeddings = model.encode(textos_raw)
            
            # Aplicar Clustering
            if algoritmo == "K-Means":
                cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
                labels = cluster_model.fit_predict(embeddings)
            elif algoritmo == "DBSCAN":
                cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = cluster_model.fit_predict(embeddings)

            # PCA para visualización
            n_components = 3 if dim_grafica == "3D" else 2
            pca = PCA(n_components=n_components)
            componentes_principales = pca.fit_transform(embeddings)

            # Armar DataFrame de resultados
            df = pd.DataFrame(componentes_principales, columns=[f"Comp {i+1}" for i in range(n_components)])
            df['Texto Original'] = textos_raw
            df['Texto Limpio'] = textos_limpios # Guardamos el texto procesado
            df['ID Clúster'] = labels
            
            # Extracción de Palabras Clave y Generación de Resúmenes
            palabras_clave_dict = extraer_palabras_clave_expandido(df, spanish_stop_words)
            
            etiquetas_cluster = []
            resumenes_html = ""
            
            for c_id in sorted(df['ID Clúster'].unique()):
                titulo, resumen = generar_titulo_y_resumen(c_id, palabras_clave_dict[c_id])
                resumenes_html += f"{resumen}\n\n"
                
                # Asignar la etiqueta a las filas correspondientes
                for idx in df[df['ID Clúster'] == c_id].index:
                    etiquetas_cluster.insert(idx, f"C{c_id}: {titulo}")
                    
            df['Tema del Clúster'] = [etiquetas_cluster[i] for i in range(len(etiquetas_cluster))]
            
            st.success("¡Análisis completado!")
            
            # --- SECCIÓN DE RESÚMENES ---
            st.markdown("### 📝 Conclusiones y Perfiles por Grupo")
            st.info(resumenes_html)
            
            # --- SECCIÓN DE GRÁFICO Y TABLA ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🌌 Gráfico de Dispersión")
                if dim_grafica == "2D":
                    fig = px.scatter(df, x='Comp 1', y='Comp 2', color='Tema del Clúster', hover_data=['Texto Original'])
                else:
                    fig = px.scatter_3d(df, x='Comp 1', y='Comp 2', z='Comp 3', color='Tema del Clúster', hover_data=['Texto Original'])
                
                fig.update_layout(legend_title_text='Grupos Temáticos', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Muestra de Datos")
                # Mostramos solo las columnas relevantes
                st.dataframe(df[['Tema del Clúster', 'Texto Original']].sort_values(by='Tema del Clúster'), hide_index=True)
                
            # Expandir detalles de limpieza (Opcional para curiosos)
            with st.expander("🔍 Ver qué texto leyó el algoritmo para nombrar los clústeres (Sin adjetivos y > 2 letras)"):
                st.dataframe(df[['Texto Original', 'Texto Limpio']])