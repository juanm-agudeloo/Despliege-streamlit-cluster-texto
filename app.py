import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Clustering de Texto con NLTK", page_icon="🧩", layout="wide")
st.title("🧩 Agrupador Automático de Textos")
st.markdown("Clustering semántico con limpieza de texto vía NLTK y nombrado dinámico de grupos.")

# --- DESCARGA Y CONFIGURACIÓN DE NLTK ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

setup_nltk()

# --- CARGA DEL MODELO DE EMBEDDINGS ---
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embeddings_model()

# --- DICCIONARIO DE FILTRADO (STOP WORDS + ADJETIVOS Y PRONOMBRES) ---
try:
    stop_words_nltk = set(stopwords.words('spanish'))
except Exception:
    stop_words_nltk = set()

# Filtro estricto: Eliminamos adjetivos, pronombres, adverbios y modificadores de cantidad
ADJETIVOS_Y_MODIFICADORES = {
    # Pronombres, determinantes y adverbios de cantidad/tiempo
    "mucho", "mucha", "muchos", "muchas", "poco", "poca", "pocos", "pocas",
    "tanto", "tanta", "tantos", "tantas", "todo", "toda", "todos", "todas",
    "uno", "una", "unos", "unas", "otro", "otra", "otros", "otras",
    "mismo", "misma", "mismos", "mismas", "cada", "algun", "alguna", "algunos", "algunas",
    "ningun", "ninguna", "ningunos", "ningunas", "nada", "algo", "alguien", "nadie",
    "siempre", "nunca", "jamas", "tambien", "tampoco", "aqui", "ahi", "alli", "alla",
    "bien", "mal", "mas", "menos", "muy", "tan", "casi", "solo", "solamente", "yo", "tu",
    
    # Adjetivos comunes (evaluativos, sentimentales, cualitativos)
    "bueno", "buena", "buenos", "buenas", "malo", "mala", "malos", "malas",
    "excelente", "excelentes", "pesimo", "pesima", "pesimos", "pesimas",
    "increible", "increibles", "terrible", "terribles", "horrible", "horribles",
    "gran", "grande", "grandes", "pequeño", "pequeña", "pequeños", "pequeñas",
    "nuevo", "nueva", "nuevos", "nuevas", "viejo", "vieja", "viejos", "viejas",
    "mejor", "mejores", "peor", "peores", "facil", "faciles", "dificil", "dificiles",
    "rapido", "rapida", "rapidos", "rapidas", "lento", "lenta", "lentos", "lentas",
    "caro", "cara", "caros", "caras", "barato", "barata", "baratos", "baratas",
    "rico", "rica", "ricos", "ricas", "delicioso", "deliciosa", "deliciosos", "deliciosas",
    "hermoso", "hermosa", "hermosos", "hermosas", "bonito", "bonita", "bonitos", "bonitas",
    "lindo", "linda", "lindos", "lindas", "agradable", "agradables", "perfecto", "perfecta",
    "inusable", "confuso", "confusa", "util", "moderno", "moderna", "limpio", "limpia"
}

PALABRAS_IGNORADAS = list(stop_words_nltk.union(ADJETIVOS_Y_MODIFICADORES))

def limpiar_texto_nltk(texto):
    """Filtra palabras de menos de 3 letras, signos y remueve adjetivos/stopwords."""
    # Extraer palabras alfabéticas de 3 o más letras
    palabras = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}\b', texto.lower())
    tokens_limpios = [p for p in palabras if p not in PALABRAS_IGNORADAS]
    return " ".join(tokens_limpios)

def extraer_palabras_clave(df):
    palabras_por_cluster = {}
    for c in df['ID Clúster'].unique():
        if c == -1:
            palabras_por_cluster[c] = ["Ruido", "Misceláneo"]
            continue
            
        textos_cluster = df[df['ID Clúster'] == c]['Texto Limpio'].tolist()
        
        if len(textos_cluster) < 2:
            palabras_por_cluster[c] = ["Específico", "Puntual"]
            continue
            
        vectorizer = TfidfVectorizer(max_features=8)
        try:
            X = vectorizer.fit_transform(textos_cluster)
            palabras = vectorizer.get_feature_names_out()
            importancias = X.sum(axis=0).A1
            palabras_ordenadas = [palabra for _, palabra in sorted(zip(importancias, palabras), reverse=True)]
            palabras_por_cluster[c] = palabras_ordenadas
        except Exception:
            palabras_por_cluster[c] = ["Indefinido", "General"]
            
    return palabras_por_cluster

def generar_titulo_y_resumen(cluster_idx, palabras):
    if cluster_idx == -1:
        titulo = "Ruido (Atípicos)"
        resumen = "🌪️ **Clúster Ruido:** Opiniones atípicas o dispersas que no concuerdan con los temas centrales."
        return titulo, resumen
    
    palabras_cap = [p.capitalize() for p in palabras]
    
    if len(palabras_cap) >= 2:
        titulo = f"{palabras_cap[0]} & {palabras_cap[1]}"
        conceptos_extra = ", ".join(palabras_cap[2:6]) if len(palabras_cap) > 2 else "temáticas afines"
        
        resumen = (f"👥 **Clúster {cluster_idx} - {titulo}:** Agrupa los textos que discuten "
                   f"principalmente sobre **{palabras_cap[0]}** y **{palabras_cap[1]}**. "
                   f"Se detectan con frecuencia términos asociados como: *{conceptos_extra}*.")
    else:
        titulo = palabras_cap[0] if palabras_cap else "General"
        resumen = f"👥 **Clúster {cluster_idx} - {titulo}:** Textos fuertemente enfocados en torno a *{titulo}*."
    
    return titulo, resumen

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    algoritmo = st.selectbox("Algoritmo de Clustering", ["K-Means", "DBSCAN"])
    if algoritmo == "K-Means":
        num_clusters = st.slider("Número de clústeres (k)", min_value=2, max_value=15, value=4)
    else:
        eps = st.slider("Distancia máxima (eps)", min_value=0.1, max_value=2.0, value=0.5)
        min_samples = st.slider("Mínimo de muestras", min_value=2, max_value=10, value=2)
        
    st.header("📊 Configuración de Gráfica")
    dim_grafica = st.radio("Dimensionalidad", ["2D", "3D"], horizontal=True)

# --- ÁREA DE ENTRADA ---
st.subheader("1. Entrada de Datos")
opcion_entrada = st.radio("¿Fuente de los datos?", ("Escribir texto", "Subir archivo (.txt o .csv)"), horizontal=True)

textos_raw = []
if opcion_entrada == "Escribir texto":
    texto_usuario = st.text_area("Ingresa los textos a analizar (uno por línea):", height=200)
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

# --- RESULTADOS ---
st.subheader("2. Resultados del Análisis")

if st.button("Ejecutar Clustering 🚀"):
    if len(textos_raw) == 0:
        st.warning("⚠️ Por favor, ingresa o sube un texto antes de ejecutar.")
    elif algoritmo == "K-Means" and len(textos_raw) < num_clusters:
        st.error(f"❌ Para {num_clusters} clústeres necesitas al menos {num_clusters} oraciones de entrada.")
    else:
        with st.spinner('Limpiando texto con NLTK, calculando embeddings y agrupando...'):
            # 1. Limpieza con NLTK para extracción semántica
            textos_limpios = [limpiar_texto_nltk(t) for t in textos_raw]
            
            # 2. Vectorización densa (oración completa para retener significado)
            embeddings = model.encode(textos_raw)
            
            # 3. Clustering
            if algoritmo == "K-Means":
                cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
                labels = cluster_model.fit_predict(embeddings)
            elif algoritmo == "DBSCAN":
                cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = cluster_model.fit_predict(embeddings)

            # 4. Reducción PCA
            n_components = 3 if dim_grafica == "3D" else 2
            pca = PCA(n_components=n_components)
            componentes = pca.fit_transform(embeddings)

            df = pd.DataFrame(componentes, columns=[f"Comp {i+1}" for i in range(n_components)])
            df['Texto Original'] = textos_raw
            df['Texto Limpio'] = textos_limpios
            df['ID Clúster'] = labels
            
            # 5. Extracción de palabras clave y resúmenes
            palabras_clave_dict = extraer_palabras_clave(df)
            
            etiquetas_cluster = {}
            resumenes_list = []
            
            for c_id in sorted(df['ID Clúster'].unique()):
                titulo, resumen = generar_titulo_y_resumen(c_id, palabras_clave_dict[c_id])
                resumenes_list.append(resumen)
                etiquetas_cluster[c_id] = f"C{c_id}: {titulo}"
                
            df['Tema del Clúster'] = df['ID Clúster'].map(etiquetas_cluster)
            
            st.success("¡Análisis completado!")
            
            # Sección de resúmenes
            st.markdown("### 📝 Conclusiones por Grupo")
            for r in resumenes_list:
                st.info(r)
            
            # Visualización
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
                st.dataframe(df[['Tema del Clúster', 'Texto Original']].sort_values(by='Tema del Clúster'), hide_index=True)
                
            with st.expander("🔍 Ver texto procesado sin adjetivos ni palabras cortas"):
                st.dataframe(df[['Texto Original', 'Texto Limpio']])
