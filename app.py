import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Clustering de Texto con K-Means", page_icon="🧩", layout="wide")
st.title("🧩 Agrupador Automático de Textos (K-Means)")
st.markdown("Clustering semántico con vectorización densa, limpieza vía NLTK y nombrado temático de grupos.")

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

ADJETIVOS_Y_MODIFICADORES = {
    # Pronombres, determinantes y adverbios
    "mucho", "mucha", "muchos", "muchas", "poco", "poca", "pocos", "pocas",
    "tanto", "tanta", "tantos", "tantas", "todo", "toda", "todos", "todas",
    "uno", "una", "unos", "unas", "otro", "otra", "otros", "otras",
    "mismo", "misma", "mismos", "mismas", "cada", "algun", "alguna", "algunos", "algunas",
    "ningun", "ninguna", "ningunos", "ningunas", "nada", "algo", "alguien", "nadie",
    "siempre", "nunca", "jamas", "tambien", "tampoco", "aqui", "ahi", "alli", "alla",
    "bien", "mal", "mas", "menos", "muy", "tan", "casi", "solo", "solamente", "yo", "tu",
    
    # Adjetivos comunes
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
    """Filtra palabras de menos de 3 letras y elimina stop words/adjetivos."""
    palabras = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}\b', texto.lower())
    tokens_limpios = [p for p in palabras if p not in PALABRAS_IGNORADAS]
    return " ".join(tokens_limpios)

def extraer_palabras_clave(df):
    """Obtiene las palabras más representativas de cada clúster con TF-IDF."""
    palabras_por_cluster = {}
    for c in sorted(df['ID Clúster'].unique()):
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
    """Genera la etiqueta dual y la explicación para el clúster."""
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
    st.header("⚙️ Parámetros de K-Means")
    num_clusters = st.slider("Número de clústeres (k)", min_value=2, max_value=15, value=4)
        
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
    elif len(textos_raw) < num_clusters:
        st.error(f"❌ Para {num_clusters} clústeres necesitas al menos {num_clusters} oraciones de entrada.")
    else:
        with st.spinner('Limpiando texto con NLTK, calculando embeddings y agrupando con K-Means...'):
            # 1. Limpieza de texto para etiquetas
            textos_limpios = [limpiar_texto_nltk(t) for t in textos_raw]
            
            # 2. Vectorización densa
            embeddings = model.encode(textos_raw)
            
            # 3. K-Means
            cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
            labels = cluster_model.fit_predict(embeddings)

            # 4. Reducción PCA para visualización
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
