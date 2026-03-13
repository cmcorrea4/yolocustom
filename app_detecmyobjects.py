# app.py — DetecMyObjects: cargador | controlador | reloj
# Sube best.pt a la raíz de tu repositorio

from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="DetecMyObjects",
    page_icon="🔍",
    layout="wide"
)

# Clases del modelo personalizado
CLASES  = ["cargador", "controlador", "reloj"]
EMOJIS  = {"cargador": "🔌", "controlador": "🎮", "reloj": "⌚"}
COLORES = ["#e94560", "#0f3460", "#533483"]

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    # Asegúrate de subir best.pt al repo
    return YOLO("best.pt")

st.title("🔍 DetecMyObjects")
st.markdown("""
Detecta automáticamente **cargadores** 🔌, **controladores** 🎮 y **relojes** ⌚
en imágenes capturadas con tu cámara.
""")

with st.spinner("Cargando modelo personalizado..."):
    model = load_model()

with st.sidebar:
    st.title("⚙️ Parámetros")
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.05)
    iou_threshold  = st.slider("Umbral IoU",       0.0, 1.0, 0.45, 0.05)
    
    st.divider()
    st.caption("**Clases detectables:**")
    for clase, emoji in EMOJIS.items():
        st.caption(f"{emoji} {clase}")

picture = st.camera_input("📸 Capturar imagen")

if picture:
    bytes_data = picture.getvalue()
    pil_img    = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    np_img     = np.array(pil_img)[..., ::-1]  # RGB → BGR para YOLO

    with st.spinner("Detectando objetos..."):
        results = model(np_img, conf=conf_threshold, iou=iou_threshold)

    result        = results[0]
    annotated_rgb = result.plot()[:, :, ::-1]  # BGR → RGB

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen con detecciones")
        st.image(annotated_rgb, use_container_width=True)

    with col2:
        st.subheader("Resultados")
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            from collections import Counter
            conteo = Counter()
            confs  = {}

            for box in boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                conteo[cls_id] += 1
                confs.setdefault(cls_id, []).append(conf)

            data = [
                {
                    "Objeto":    f"{EMOJIS[CLASES[i]]} {CLASES[i]}",
                    "Cantidad":  cnt,
                    "Confianza": f"{np.mean(confs[i]):.1%}"
                }
                for i, cnt in conteo.items()
            ]

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Objeto")["Cantidad"])
        else:
            st.info("No se detectaron objetos.")
            st.caption("Prueba a reducir el umbral de confianza.")

st.divider()
st.caption("DetecMyObjects · YOLOv11 · Entrenado con Roboflow · 3 clases")