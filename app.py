import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import smtplib
import base64
import os
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pandas.tseries.offsets import BusinessDay
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import time
from fpdf import FPDF
# =====================================================================
# CONSTANTES Y CONFIGURACIÓN INICIAL
# =====================================================================
CARRERAS_OFICIALES = [
    "Abogacía", 
    "Licenciatura en Trabajo Social", 
    "Licenciatura en Ciencias Políticas", 
    "Licenciatura en Historia", 
    "Licenciatura en Comunicación"
]

STOP_WORDS_ES = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", 
    "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque",
    "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso"
]

st.set_page_config(page_title="Nexo Memoria - UNMa", page_icon="🎗️", layout="wide")

# Custom CSS Premium UNMa
st.markdown("""
<style>
    :root {
        --unma-blue: #1a365d;
        --unma-white: #ffffff;
        --unma-gray: #f4f6f9;
        --unma-accent: #c4a152; 
    }
    
    /* Header Principal */
    .unma-header {
        background-color: var(--unma-blue);
        color: var(--unma-white);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 6px 15px rgba(26, 54, 93, 0.25);
        display: flex;
        align-items: center;
        gap: 20px;
        position: relative;
        overflow: hidden;
    }
    
    /* Símbolo de Pañuelo de Fondo Sutil */
    .unma-header::after {
        content: '🎗️';
        position: absolute;
        right: 10%;
        top: -20px;
        font-size: 10rem;
        opacity: 0.05;
        pointer-events: none;
    }

    .unma-header h1 {
        margin: 0;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--unma-white);
        letter-spacing: 0.5px;
    }
    
    /* Kanban Cards */
    .kanban-card {
        background-color: var(--unma-white);
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e4e8;
        border-left: 6px solid var(--unma-blue);
        color: #333333;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
    }
    .kanban-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .kanban-card h4 {
        margin-top: 0;
        color: var(--unma-blue);
        font-weight: 700;
        margin-bottom: 5px;
        font-size: 1.1rem;
    }
    
    /* Alertas */
    .alert-urgent {
        background-color: #d32f2f;
        color: white;
        padding: 6px 10px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 12px;
        font-size: 0.8rem;
        box-shadow: 0 2px 5px rgba(211,47,47,0.3);
    }
    
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Encabezado UNMa
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "logo_unma.png"
if os.path.exists(logo_path):
    logo_base64 = get_base64_of_bin_file(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="max-height: 80px; max-width: 200px;">'
else:
    logo_html = '<span style="font-size: 3.5rem;">🎗️</span>'

st.markdown(f"""
<div class="unma-header">
    <div style="background: white; padding: 15px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        {logo_html}
    </div>
    <div>
        <h1>Nexo Memoria</h1>
        <p style="margin: 0; opacity: 0.9; font-size: 1.2rem; font-weight: 300;">Sistema Universitario de Comunicación Interna — <b>UNMa</b></p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# CONEXIÓN A GOOGLE SHEETS
# =====================================================================
@st.cache_resource(ttl=300) # ttl reducido para refresco más frecuente de la base
def init_connection():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)

def get_sheet():
    return init_connection().open("Nexo_Memoria_DB") 

def get_data(sheet_name):
    try:
        sheet = get_sheet().worksheet(sheet_name)
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        return pd.DataFrame()

# =====================================================================
# MODELO NLP AVANZADO (LinearSVC + Bi-gramas)
# =====================================================================
@st.cache_resource(ttl=3600)
def train_nlp_models(df):
    """Entrena modelos LinearSVC robustos con Fallbacks para nombres de columnas."""
    if df.empty or 'Asunto' not in df.columns:
        return None, None, None, None, 'Detalle'
        
    # DETECCIÓN DE COLUMNA DETALLE
    col_detalle = 'Detalle (indique Carrera si corresponde)' if 'Detalle (indique Carrera si corresponde)' in df.columns else 'Detalle'
    if col_detalle not in df.columns:
        return None, None, None, None, 'Detalle' # Si no hay ni Detalle ni Detalle largo
        
    # Limpiar NA
    df_train = df.dropna(subset=['Asunto', col_detalle, 'Tipo de Comunicación', 'Tema'])
    
    # Aumentar sensibilidad: entrenar incluso si hay 2 registros (antes 5)
    if len(df_train) < 2:
        return None, None, None, None, col_detalle

    X_text = df_train['Asunto'].astype(str) + " " + df_train[col_detalle].astype(str)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=STOP_WORDS_ES)
    X = vectorizer.fit_transform(X_text)
    
    clf_tipo = LinearSVC(random_state=42)
    clf_tipo.fit(X, df_train['Tipo de Comunicación'])
    
    clf_tema = LinearSVC(random_state=42)
    clf_tema.fit(X, df_train['Tema'])
    
    clf_prio = None
    if 'Prioridad' in df_train.columns:
        df_train_prio = df_train.dropna(subset=['Prioridad'])
        if len(df_train_prio) >= 2:
            clf_prio = LinearSVC(random_state=42)
            clf_prio.fit(X[df_train_prio.index.isin(df_train.index)], df_train_prio['Prioridad'])
    
    return vectorizer, clf_tipo, clf_tema, clf_prio, col_detalle

def predict_nlp(asunto, detalle, vectorizer, clf_tipo, clf_tema, clf_prio):
    """Realiza la predicción."""
    if not vectorizer or not asunto:
        return None, None, None, None
    X_new = vectorizer.transform([asunto + " " + detalle])
    pred_tipo = clf_tipo.predict(X_new)[0] if clf_tipo else None
    pred_tema = clf_tema.predict(X_new)[0] if clf_tema else None
    pred_prio = clf_prio.predict(X_new)[0] if clf_prio else None
    
    recomendacion = "Revisar manualmente."
    if pred_tema in ["Presupuesto", "Infraestructura"]:
        recomendacion = "Sugerimos derivar a Admin o Secretaria Académica."
    elif pred_tema == "Gestión Alumnos":
        recomendacion = "Sugerimos derivar a Director de Carrera."
        
    return pred_tipo, pred_tema, pred_prio, recomendacion

# =====================================================================
# SISTEMA DE ALERTAS Y LÓGICA DE NEGOCIO
# =====================================================================
def es_vencimiento_proximo(fecha_limite_str):
    if not fecha_limite_str:
        return False
    try:
        limite = pd.to_datetime(fecha_limite_str).date()
        hoy = datetime.now().date()
        proximo_habil = (hoy + BusinessDay(1)).date()
        # Consideramos próximo o si ya venció (y sigue pendiente)
        return limite <= proximo_habil
    except:
        return False

def enviar_notificacion(destinatarios_bcc, asunto, mensaje):
    try:
        user = st.secrets["email_user"]
        password = st.secrets["email_password"]
        
        msg = MIMEText(mensaje)
        msg['Subject'] = asunto
        msg['From'] = user
        
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(user, password)
        server.sendmail(user, destinatarios_bcc, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        return False

# =====================================================================
# ACTUALIZACIÓN DE KANBAN BIDIRECCIONAL
# =====================================================================
def update_task_status(row_index, nuevo_estado, respuesta=""):
    try:
        sheet = get_sheet().worksheet("BaseDatos")
        sheet.update_cell(row_index + 2, 10, nuevo_estado) # Col J: Estado
        
        if respuesta:
            sheet.update_cell(row_index + 2, 13, respuesta) # Col M: Respuesta
        
        if nuevo_estado in ["Resuelto", "Informado"]:
            fecha_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet.update_cell(row_index + 2, 15, fecha_fin) # Col O: Fecha_Fin
        
        # Limpiar caché forzadamente para que el refresco muestre datos nuevos
        st.cache_resource.clear()
        st.rerun() 
    except Exception as e:
        st.error(f"Error de sincronización: {e}")

# =====================================================================
# SISTEMA DE INFORMES PDF
# =====================================================================
def generar_pdf(df, titulo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, txt=f"Nexo Memoria UNMa - {titulo}", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Fecha de reporte: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    cols_to_print = [c for c in ['Asunto', 'Estado', 'Prioridad', 'Aula'] if c in df.columns]
    col_widths = [70, 40, 30, 40]
    
    # Header
    pdf.set_font("Arial", size=10, style='B')
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    
    for i, col in enumerate(cols_to_print):
        w = col_widths[i] if i < len(col_widths) else 40
        pdf.cell(w, 10, str(col), border=1, fill=True)
    pdf.ln()
    
    # Rows
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(0, 0, 0)
    
    for _, row in df.iterrows():
        for i, col in enumerate(cols_to_print):
            w = col_widths[i] if i < len(col_widths) else 40
            text = str(row[col])[:45] 
            # Eliminar caracteres raros si los hay para evitar error de FPDF con iso-8859-1
            text = text.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(w, 10, text, border=1)
        pdf.ln()
        
    return bytes(pdf.output())

# =====================================================================
# CALENDARIO INSTITUCIONAL (VISUAL)
# =====================================================================
def render_calendario(df_visible):
    """Renderiza tarjetas de calendario organizadas por día."""
    st.markdown("### 📅 Calendario de Vencimientos Institucionales")
    
    search_query = st.text_input("🔍 Buscar en calendario (por asunto o detalle)...", "")
    
    df_activos = df_visible[
        (~df_visible['Estado'].isin(['Resuelto', 'Informado'])) & 
        (df_visible['Fecha_Limite'] != "")
    ].copy()
    
    if search_query:
        df_activos = df_activos[
            df_activos['Asunto'].str.contains(search_query, case=False, na=False) |
            df_activos.get('Detalle', pd.Series(dtype=str)).str.contains(search_query, case=False, na=False)
        ]
    
    if df_activos.empty:
        st.info("No hay tareas urgentes programadas con fecha de vencimiento.")
        return
        
    try:
        col_btn_pdf, _ = st.columns([1, 3])
        with col_btn_pdf:
            pdf_bytes = generar_pdf(df_activos, "Reporte de Calendario")
            st.download_button(
                label="📥 Descargar Informe PDF",
                data=pdf_bytes,
                file_name="reporte_calendario_unma.pdf",
                mime="application/pdf",
                key="btn_pdf_cal"
            )
            
        df_activos['Fecha_Limite_DT'] = pd.to_datetime(df_activos['Fecha_Limite'], errors='coerce')
        df_activos = df_activos.dropna(subset=['Fecha_Limite_DT']).sort_values('Fecha_Limite_DT')
        
        # Agrupar por fecha
        agrupado = df_activos.groupby(df_activos['Fecha_Limite_DT'].dt.date)
        
        for fecha, grupo in agrupado:
            # Check si es urgente
            urgente = es_vencimiento_proximo(str(fecha))
            
            # Estructura de tarjeta de calendario
            with st.container():
                st.markdown(f"""
                <div style="display: flex; background: white; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden;">
                    <div style="background: {'#d32f2f' if urgente else '#1a365d'}; color: white; padding: 15px; min-width: 80px; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                        <span style="font-size: 1.5rem; font-weight: bold;">{fecha.day}</span>
                        <span style="font-size: 0.8rem; text-transform: uppercase;">{fecha.strftime('%b')}</span>
                    </div>
                    <div style="padding: 15px; flex-grow: 1;">
                """, unsafe_allow_html=True)
                
                for _, row in grupo.iterrows():
                    with st.expander(f"📌 {row['Asunto']} - {row.get('Aula', 'N/A')}"):
                        st.markdown(f"**Asignado a:** {row.get('Destinatario Rol', 'N/A')}")
                        st.markdown(f"**Detalle:** {row.get('Detalle', 'Sin detalle')}")
                        enlaces = row.get('Enlaces_Referencias', '')
                        if pd.notna(enlaces) and str(enlaces).strip():
                            st.markdown("**🔗 Enlaces Adjuntos:**")
                            for link in str(enlaces).split('\\n'):
                                if link.strip():
                                    st.markdown(f"- [{link.strip()}]({link.strip()})")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
    except Exception as e:
        st.warning(f"No se pudo renderizar el calendario correctamente: {e}")

# =====================================================================
# HERRAMIENTAS DE IA (GENERADOR DE PROMPTS)
# =====================================================================
def stream_data(text):
    """Generador para simular el efecto de escritura de IA."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

def render_ia_helpers(prompt_texto, animate=False):
    """Renderiza botones de IA y un prompt copiable con animación opcional."""
    st.markdown("**Prompt sugerido:**")
    st.caption("✅ Usa el ícono de las dos hojitas arriba a la derecha del recuadro para **Copiar el Prompt** instantáneamente.")
    
    if animate:
        st.write_stream(stream_data(prompt_texto))
        
    st.code(prompt_texto, language='markdown')
    
    st.markdown("""
    <div style="display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; margin-bottom: 15px;">
        <a href="https://chatgpt.com/" target="_blank" style="text-decoration: none; background-color: #10a37f; color: white; padding: 8px 15px; border-radius: 5px; font-weight: bold; font-size: 0.9em;">🤖 ChatGPT</a>
        <a href="https://gemini.google.com/" target="_blank" style="text-decoration: none; background-color: #4285f4; color: white; padding: 8px 15px; border-radius: 5px; font-weight: bold; font-size: 0.9em;">✨ Gemini</a>
        <a href="https://claude.ai/" target="_blank" style="text-decoration: none; background-color: #d97757; color: white; padding: 8px 15px; border-radius: 5px; font-weight: bold; font-size: 0.9em;">🧠 Claude</a>
        <div title="En entrenamiento con base de datos de Nexo Memoria" style="background-color: #cccccc; color: #666666; padding: 8px 15px; border-radius: 5px; font-weight: bold; font-size: 0.9em; cursor: not-allowed; opacity: 0.7;">🏛️ Modelo UNMa (Próximamente)</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# LÓGICA DE LOGIN Y APP PRINCIPAL
# =====================================================================
def login_user():
    st.subheader("Acceso al Sistema")
    with st.form("login_form"):
        email = st.text_input("Usuario (Email)")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Ingresar")
        
        if submit:
            usuarios_df = get_data("Usuarios")
            if not usuarios_df.empty and 'Usuario' in usuarios_df.columns:
                usuarios_df['Usuario'] = usuarios_df['Usuario'].astype(str).str.strip()
                usuarios_df['Contraseña'] = usuarios_df['Contraseña'].astype(str).str.strip()
                user_row = usuarios_df[(usuarios_df['Usuario'] == email.strip()) & (usuarios_df['Contraseña'] == password.strip())]
                
                if not user_row.empty:
                    st.session_state['logged_in'] = True
                    st.session_state['user_info'] = {
                        'email': email,
                        'rol_carrera': user_row.iloc[0]['Rol_Carrera'],
                        'nombre': user_row.iloc[0]['Nombre_Completo']
                    }
                    st.rerun()
                else:
                    st.error("Credenciales inválidas.")

def main():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login_user()
        return

    # === TIEMPO REAL: Auto-Refresh cada 30 segundos ===
    # El usuario no notará recargas bruscas de la ventana, la caché y estados fluirán solos.
    st_autorefresh(interval=30000, limit=None, key="data_refresh_timer")

    user_info = st.session_state['user_info']
    rol_general = user_info['rol_carrera'].split(":")[0].strip()
    carrera_acceso = user_info['rol_carrera'].split(":")[1].strip() if ":" in user_info['rol_carrera'] else "Todas"
    
    st.sidebar.markdown("### 👤 Perfil Activo")
    st.sidebar.write(f"**Nombre:** {user_info['nombre']}")
    st.sidebar.write(f"🛡️ **Rol:** {rol_general}")
    st.sidebar.write(f"🎓 **Alcance:** {carrera_acceso}")
    
    # Botón manual de refresco por si acaso
    if st.sidebar.button("🔄 Refrescar Tablero", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
        
    st.sidebar.divider()
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['logged_in'] = False
        st.rerun()

    df_main = get_data("BaseDatos")
    
    if df_main.empty:
        st.info("La base de datos está vacía. ¡Ve a la pestaña 'Cargar Solicitud' para registrar la primera!")
        columnas_requeridas = [
            'Fecha', 'Tipo de Comunicación', 'Tema', 'Asunto', 'Detalle', 'Detalle (indique Carrera si corresponde)',
            'Remitente ROL', 'RemitenteNombre y Apellido', 'Destinatario Rol', 'Destinatario Nombre y Apellido',
            'Estado', 'Aula', 'Docente', 'Respuesta (si fuera necesario)', 'Respuesta', 'Fecha_Inicio', 'Fecha_Fin',
            'Fecha_Limite', 'Prioridad'
        ]
        df_main = pd.DataFrame(columns=columnas_requeridas)

    # Fallbacks de columnas vitales
    col_respuesta = 'Respuesta (si fuera necesario)' if 'Respuesta (si fuera necesario)' in df_main.columns else ('Respuesta' if 'Respuesta' in df_main.columns else None)
    
    # Filtrar según rol de acceso
    if rol_general not in ["Admin", "Secretaria Academica"] and carrera_acceso != "Todas":
        df_visible = df_main[df_main['Aula'] == carrera_acceso]
    else:
        df_visible = df_main

    # Renderizar Agenda en Sidebar (mini version)
    with st.sidebar:
        st.markdown("---")
        st.caption("🟢 Conectado - AutoSync Activado (30s)")

    # Estructura de Pestañas
    if rol_general in ["Admin", "Secretaria Academica"]:
        pestañas = ["📊 Tablero de Operaciones", "➕ Cargar Solicitud", "📅 Calendario", "🤖 Taller de Prompts", "📈 Dashboard"]
    else:
        pestañas = ["📊 Tablero de Operaciones", "➕ Cargar Solicitud", "📅 Calendario", "🤖 Taller de Prompts"]
        
    tabs = st.tabs(pestañas)

    if st.session_state.get('switch_to_tablero', False):
        import streamlit.components.v1 as components
        components.html("""
            <script>
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length > 0) {
                    tabs[0].click();
                }
            </script>
        """, height=0, width=0)
        st.session_state['switch_to_tablero'] = False

    with tabs[0]: # TABLERO KANBAN
        if 'Aula' in df_visible.columns:
            todas_carreras = df_visible['Aula'].unique()
            carreras_presentes_oficiales = [c for c in todas_carreras if c in CARRERAS_OFICIALES]
            carreras_otras = [c for c in todas_carreras if c not in CARRERAS_OFICIALES and str(c).strip() != ""]
            
            pestañas_generar = carreras_presentes_oficiales.copy()
            if carreras_otras:
                pestañas_generar.append("Otras / Institucional")
                
            if not pestañas_generar:
                st.info("Aún no hay tareas activas en el tablero para tu área.")
            else:
                tabs_carreras = st.tabs(pestañas_generar)
                
                for i, pest_name in enumerate(pestañas_generar):
                    with tabs_carreras[i]:
                        if pest_name == "Otras / Institucional":
                            df_pest = df_visible[df_visible['Aula'].isin(carreras_otras)]
                        else:
                            df_pest = df_visible[df_visible['Aula'] == pest_name]
                            
                        # === DESCARGA DE INFORME ===
                        if not df_pest.empty:
                            pdf_bytes = generar_pdf(df_pest, f"Reporte de Estado - {pest_name}")
                            st.download_button(
                                label=f"📥 Descargar Informe PDF de {pest_name}",
                                data=pdf_bytes,
                                file_name=f"reporte_{pest_name}.pdf".replace(" ", "_"),
                                mime="application/pdf",
                                key=f"btn_pdf_{i}"
                            )
                            
                        # === RENDER KANBAN BIDIRECCIONAL ===
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### 🔴 En Desarrollo")
                            for idx, row in df_pest[df_pest['Estado'] == 'Pendiente'].iterrows():
                                with st.container():
                                    st.markdown('<div class="kanban-card">', unsafe_allow_html=True)
                                    if es_vencimiento_proximo(row.get('Fecha_Limite', '')):
                                        st.markdown('<div class="alert-urgent">🚨 ALERTA UNMa: Vence el próximo día hábil</div>', unsafe_allow_html=True)
                                    
                                    st.markdown(f"<h4>{row.get('Asunto','')}</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<p><strong>{row.get('Prioridad', 'Media')}</strong> | {row.get('Tipo de Comunicación', '')}</p>", unsafe_allow_html=True)
                                    st.caption(f"De: {row.get('RemitenteNombre y Apellido', '')}")
                                    
                                    if rol_general in ["Admin", "Secretaria Academica", "Director"]:
                                        if st.button("Tomar Caso ➔", key=f"btn_tomar_{idx}"):
                                            update_task_status(idx, "En proceso")
                                    st.markdown('</div>', unsafe_allow_html=True)

                        with col2:
                            st.markdown("#### 🟡 En Proceso")
                            for idx, row in df_pest[df_pest['Estado'] == 'En proceso'].iterrows():
                                with st.container():
                                    st.markdown('<div class="kanban-card" style="border-left-color: #f5a623;">', unsafe_allow_html=True)
                                    st.markdown(f"<h4>{row.get('Asunto','')}</h4>", unsafe_allow_html=True)
                                    
                                    col_btn1, col_btn2 = st.columns(2)
                                    with col_btn1:
                                        if rol_general in ["Admin", "Secretaria Academica", "Director"]:
                                            if st.button("⬅️ Devolver", key=f"btn_dev_{idx}"):
                                                update_task_status(idx, "Pendiente")
                                    
                                    if rol_general in ["Admin", "Secretaria Academica", "Director"]:
                                        with st.expander("✨ Ayuda IA para redactar"):
                                            if st.button("Generar Borrador IA", key=f"btn_ia_res_{idx}"):
                                                detalle_texto = row.get('Detalle (indique Carrera si corresponde)') or row.get('Detalle') or ''
                                                prompt_resolucion = f"Actúa como un directivo académico de la Universidad Nacional de Madres de Plaza de Mayo (UNMa) y redacta una respuesta formal, empática y resolutiva para la siguiente situación académica:\\n\\nAsunto: {row.get('Asunto', '')}\\nDetalle: {detalle_texto}\\n\\nAsegúrate de reflejar los valores institucionales de la Universidad."
                                                render_ia_helpers(prompt_resolucion, animate=True)
                                            
                                        respuesta = st.text_area("Resolución", key=f"resp_{idx}", placeholder="Dictamen final...")
                                        if st.button("Finalizar ✅", key=f"btn_fin_{idx}", use_container_width=True):
                                            update_task_status(idx, "Resuelto", respuesta)
                                    elif rol_general == "Articuladora":
                                        st.info("Solo lectura (Jerarquía superior resuelve).")
                                        
                                    st.markdown('</div>', unsafe_allow_html=True)

                        with col3:
                            st.markdown("#### 🟢 Terminado")
                            for idx, row in df_pest[df_pest['Estado'].isin(['Resuelto', 'Informado'])].iterrows():
                                with st.container():
                                    st.markdown('<div class="kanban-card" style="border-left-color: #7ed321; opacity: 0.85;">', unsafe_allow_html=True)
                                    st.markdown(f"<h4>{row.get('Asunto','')}</h4>", unsafe_allow_html=True)
                                    
                                    resp_text = row.get(col_respuesta, 'Sin comentarios') if col_respuesta else 'Sin comentarios'
                                    st.markdown(f"<p><strong>Resolución:</strong> {resp_text}</p>", unsafe_allow_html=True)
                                    
                                    try:
                                        inicio = pd.to_datetime(row['Fecha_Inicio'])
                                        fin = pd.to_datetime(row['Fecha_Fin'])
                                        duracion = fin - inicio
                                        st.caption(f"⏱️ Resuelto en: {duracion.days}d {duracion.seconds // 3600}h")
                                    except:
                                        pass
                                    st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]: # TAB NUEVA ENTRADA
        st.markdown("### Registrar Nueva Intervención")
        
        nlp_res = train_nlp_models(df_main)
        vectorizer, clf_tipo, clf_tema, clf_prio, col_detalle_name = nlp_res if nlp_res else (None, None, None, None, 'Detalle')
        
        with st.form("new_entry_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                asunto = st.text_input("Asunto (Título corto)")
                aula = st.selectbox("Carrera / Aula", CARRERAS_OFICIALES + ["Otra área (Especificar)"])
                if aula == "Otra área (Especificar)":
                    aula = st.text_input("Especifique el área:")
                    
                docente = st.text_input("Docente o Persona Involucrada")
                prioridad = st.selectbox("Prioridad", ["Alta", "Media", "Baja"], index=1)
                fecha_limite = st.date_input("Fecha Límite (Opcional)")
            
            with col_b:
                detalle = st.text_area("Detalle (Describa la situación)", height=130)
                enlaces_text = st.text_area("🔗 Enlaces/Referencias adicionales (Pegue aquí links a Drive, YouTube, etc. Uno por línea)", height=68)
                
                with st.expander("✨ Mejorar redacción con IA"):
                    st.caption("Escribe un borrador arriba y usa estos enlaces para que la IA lo mejore institucionalmente.")
                    prompt_mejora = f"Actúa como corrector de estilo de la Universidad Nacional de Madres de Plaza de Mayo (UNMa). Por favor, mejora la redacción, corrige la gramática y dale un tono fuertemente institucional, claro y profesional al siguiente reporte:\\n\\n{detalle if detalle else '[Pega aquí tu borrador]'}"
                    render_ia_helpers(prompt_mejora, animate=False)
                    
                pred_tipo, pred_tema, pred_prio, recomendacion = predict_nlp(asunto, detalle, vectorizer, clf_tipo, clf_tema, clf_prio)
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #c4a152;">
                🤖 <strong>Sugerencias de IA:</strong><br>
                Tipo: <em>{pred_tipo or 'Sin datos'}</em> | Tema: <em>{pred_tema or 'Sin datos'}</em> | Prio. Sugerida: <em>{pred_prio or 'Sin datos'}</em><br>
                💡 <em>{recomendacion}</em>
                </div>
                """, unsafe_allow_html=True)
                
                opciones_tipo = ["Intervención", "Evento", "Consulta"]
                idx_tipo = opciones_tipo.index(pred_tipo) if pred_tipo in opciones_tipo else 0
                tipo_comunicacion = st.selectbox("Tipo de Comunicación", opciones_tipo, index=idx_tipo)
                
                opciones_tema = ["Gestión Alumnos", "Infraestructura", "Presupuesto", "Académico", "Otros"]
                idx_tema = opciones_tema.index(pred_tema) if pred_tema in opciones_tema else 0
                tema = st.selectbox("Tema", opciones_tema, index=idx_tema)
                
                dest_rol = st.selectbox("Destinatario Rol Asignado", ["Articuladora", "Director", "Secretaria Academica", "Admin"])

            submitted = st.form_submit_button("Guardar y Enviar", use_container_width=True)
            
            if submitted:
                if not asunto or not detalle:
                    st.error("Asunto y Detalle son obligatorios.")
                else:
                    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
                    fecha_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    try:
                        sheet = get_sheet().worksheet("BaseDatos")
                        headers = sheet.row_values(1)
                        
                        # Mapeo dinámico para que no importe el orden de las columnas en Google Sheets
                        datos_a_guardar = {
                            'Fecha': fecha_hoy,
                            'Tipo de Comunicación': tipo_comunicacion,
                            'Tema': tema,
                            'Asunto': asunto,
                            'Detalle': detalle,
                            'Detalle (indique Carrera si corresponde)': '',
                            'Remitente ROL': rol_general,
                            'RemitenteNombre y Apellido': user_info['nombre'],
                            'Destinatario Rol': dest_rol,
                            'Destinatario Nombre y Apellido': '',
                            'Estado': 'Pendiente',
                            'Aula': aula,
                            'Docente': docente,
                            'Respuesta (si fuera necesario)': '',
                            'Respuesta': '',
                            'Fecha_Inicio': fecha_inicio,
                            'Fecha_Fin': '',
                            'Fecha_Limite': str(fecha_limite),
                            'Prioridad': prioridad,
                            'Enlaces_Referencias': enlaces_text
                        }
                        
                        nueva_fila = [datos_a_guardar.get(h, "") for h in headers]
                        
                        sheet.append_row(nueva_fila)
                        st.success("¡Registro guardado correctamente! Actualizando...")
                        
                        usuarios_df = get_data("Usuarios")
                        if not usuarios_df.empty:
                            emails_bcc = usuarios_df[usuarios_df['Rol_Carrera'].str.contains(aula, na=False)]['Usuario'].tolist()
                            if emails_bcc:
                                enviar_notificacion(
                                    destinatarios_bcc=emails_bcc,
                                    asunto=f"Nexo Memoria UNMa: {prioridad.upper()} - {asunto}",
                                    mensaje=f"Se ha registrado una nueva tarea.\nÁrea: {aula}\nDetalle: {detalle}"
                                )
                        st.cache_resource.clear()
                        st.session_state['switch_to_tablero'] = True
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")

    with tabs[2]: # CALENDARIO
        render_calendario(df_visible)

    with tabs[3]: # TALLER DE PROMPTS
        st.markdown("### 🤖 Taller de Prompts (Generador IA)")
        st.write("Selecciona una tarea para generar un prompt optimizado y utilizarlo en tu asistente de IA preferido.")
        
        tarea_prompt = st.selectbox("¿Qué necesitas redactar?", [
            "Respuesta formal a estudiante",
            "Elevar pedido al rectorado",
            "Sintetizar reclamo complejo",
            "Notificación de evento académico"
        ])
        
        contexto_prompt = st.text_area("Contexto o ideas principales", placeholder="Ej: Hubo un problema con la inscripción y quiero avisar que se extendió el plazo...")
        
        if st.button("Generar Prompt ✨"):
            if not contexto_prompt:
                st.warning("Por favor, ingresa algo de contexto para generar el prompt.")
            else:
                if tarea_prompt == "Respuesta formal a estudiante":
                    p_final = f"Actúa como representante de la Universidad Nacional de Madres de Plaza de Mayo (UNMa). Escribe una respuesta formal y empática dirigida a un estudiante. El mensaje debe comunicar lo siguiente: {contexto_prompt}. Usa un tono institucional pero accesible y humano."
                elif tarea_prompt == "Elevar pedido al rectorado":
                    p_final = f"Actúa como directivo de la Universidad Nacional de Madres de Plaza de Mayo (UNMa). Redacta una nota formal dirigida al rectorado solicitando o informando lo siguiente: {contexto_prompt}. Debe ser conciso, justificado y mantener un alto protocolo universitario."
                elif tarea_prompt == "Sintetizar reclamo complejo":
                    p_final = f"Actúa como analista de gestión de la Universidad Nacional de Madres de Plaza de Mayo (UNMa). Resume el siguiente texto en 3 viñetas clave (Problema principal, Afectados, Urgencia) para que sea fácil de leer por las autoridades: {contexto_prompt}"
                else:
                    p_final = f"Actúa como comunicador de la Universidad Nacional de Madres de Plaza de Mayo (UNMa). Redacta una notificación clara sobre un evento académico con los siguientes detalles: {contexto_prompt}. Incluye fecha, lugar y a quién está dirigido."
                    
                render_ia_helpers(p_final, animate=True)

    if rol_general in ["Admin", "Secretaria Academica"] and len(tabs) > 4:
        with tabs[4]: # DASHBOARD ADMIN
            st.markdown("### 📈 Panel de Estadísticas y Control")
            
            if not df_visible.empty and 'Estado' in df_visible.columns:
                col_chart1, col_chart2 = st.columns(2)
                col_chart3, col_chart4 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Distribución de Estados**")
                    estado_counts = df_visible['Estado'].value_counts().reset_index()
                    estado_counts.columns = ['Estado', 'Cantidad']
                    fig_estados = px.pie(estado_counts, values='Cantidad', names='Estado', hole=0.4, 
                                         color='Estado',
                                         color_discrete_map={'Pendiente':'#d32f2f', 'En proceso':'#f5a623', 'Resuelto':'#7ed321', 'Informado':'#7ed321'})
                    st.plotly_chart(fig_estados, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Casos por Carrera/Área**")
                    carrera_counts = df_visible['Aula'].value_counts().reset_index()
                    carrera_counts.columns = ['Carrera', 'Cantidad']
                    fig_carreras = px.bar(carrera_counts, x='Carrera', y='Cantidad', 
                                          color_discrete_sequence=['#1a365d'])
                    st.plotly_chart(fig_carreras, use_container_width=True)
                    
                with col_chart3:
                    st.markdown("**Evolución Temporal de Solicitudes**")
                    if 'Fecha' in df_visible.columns:
                        try:
                            df_fechas = df_visible.copy()
                            df_fechas['Fecha'] = pd.to_datetime(df_fechas['Fecha']).dt.date
                            fecha_counts = df_fechas.groupby('Fecha').size().reset_index(name='Cantidad')
                            fig_fechas = px.line(fecha_counts, x='Fecha', y='Cantidad', markers=True, color_discrete_sequence=['#c4a152'])
                            st.plotly_chart(fig_fechas, use_container_width=True)
                        except:
                            st.info("No se puede generar gráfico temporal con los datos actuales.")
                            
                with col_chart4:
                    st.markdown("**Prioridad de Solicitudes Activas**")
                    if 'Prioridad' in df_visible.columns:
                        activos = df_visible[~df_visible['Estado'].isin(['Resuelto', 'Informado'])]
                        if not activos.empty:
                            prio_counts = activos['Prioridad'].value_counts().reset_index()
                            prio_counts.columns = ['Prioridad', 'Cantidad']
                            fig_prio = px.pie(prio_counts, values='Cantidad', names='Prioridad',
                                            color_discrete_map={'Alta':'#d32f2f', 'Media':'#f5a623', 'Baja':'#7ed321'})
                            st.plotly_chart(fig_prio, use_container_width=True)
                        else:
                            st.info("No hay solicitudes activas para analizar prioridad.")
                
                # Mapa de Calor
                st.markdown("---")
                st.markdown("**Mapa de Calor: Carrera vs Estado**")
                try:
                    heatmap_data = pd.crosstab(df_visible['Aula'], df_visible['Estado'])
                    fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", 
                                       color_continuous_scale=["#7ed321", "#f5a623", "#d32f2f"],
                                       labels=dict(x="Estado", y="Carrera/Área", color="Cantidad"))
                    st.plotly_chart(fig_heat, use_container_width=True)
                except Exception as e:
                    st.warning("No se pudo generar el mapa de calor.")
            else:
                st.info("No hay datos suficientes para graficar.")

if __name__ == "__main__":
    main()
