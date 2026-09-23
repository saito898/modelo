import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="💜",
    layout="wide"
)


# =========================================================
# FONDO Y ESTILOS
# =========================================================

# Cargar imagen de fondo
background_image = Image.open("animetech.avif")

# Convertir la imagen a base64 para usarla como fondo
import base64
from io import BytesIO

buffered = BytesIO()
background_image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()


st.markdown(
    f"""
    <style>

    /* =====================================================
       FONDO DE LA PÁGINA
       ===================================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(43, 20, 70, 0.78),
                rgba(76, 32, 90, 0.72)
            ),
            url("data:image/png;base64,{img_base64}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}


    /* =====================================================
       CONTENEDOR PRINCIPAL
       ===================================================== */

    .block-container {{
        max-width: 1100px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }}


    /* =====================================================
       TITULO PRINCIPAL
       ===================================================== */

    h1 {{
        color: #FFD166 !important;
        font-size: 46px !important;
        font-weight: 800 !important;
        text-align: center;

        text-shadow:
            0 3px 12px rgba(0, 0, 0, 0.45);
    }}


    /* =====================================================
       SUBTITULOS
       ===================================================== */

    h2 {{
        color: #FFD166 !important;
        font-weight: 800 !important;

        text-shadow:
            0 2px 8px rgba(0, 0, 0, 0.35);
    }}

    h3 {{
        color: #FF9F1C !important;
        font-weight: 750 !important;
    }}


    /* =====================================================
       TEXTO
       ===================================================== */

    p {{
        color: #ffffff;
        font-size: 16px;

        text-shadow:
            0 2px 5px rgba(0, 0, 0, 0.4);
    }}


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                rgba(57, 25, 91, 0.98),
                rgba(103, 44, 120, 0.98)
            );

        border-right: 2px solid #FF9F1C;
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFD166 !important;
    }}


    /* =====================================================
       CAMARA
       ===================================================== */

    [data-testid="stCameraInput"] {{
        background: rgba(255, 255, 255, 0.96);

        border-radius: 24px;

        padding: 15px;

        border: 3px solid #FF9F1C;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.35);
    }}


    /* =====================================================
       IMAGEN MOSTRADA
       ===================================================== */

    [data-testid="stImage"] {{
        background: rgba(255, 255, 255, 0.95);

        padding: 12px;

        border-radius: 24px;

        border: 3px solid #FFD166;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.35);
    }}


    /* =====================================================
       MENSAJES DE RESULTADO
       ===================================================== */

    [data-testid="stAlert"] {{
        border-radius: 18px;
    }}


    /* =====================================================
       DIVISORES
       ===================================================== */

    hr {{
        border: none;
        height: 2px;

        background:
            linear-gradient(
                to right,
                transparent,
                #FF9F1C,
                #FFD166,
                #FF9F1C,
                transparent
            );

        margin: 2rem 0;
    }}


    /* =====================================================
       BOTONES
       ===================================================== */

    button {{
        border-radius: 12px !important;
    }}


    /* =====================================================
       ESPACIO PARA EL TITULO
       ===================================================== */

    [data-testid="stTitle"] {{
        margin-bottom: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# VERSIÓN DE PYTHON
# =========================================================

st.write(
    "Versión de Python:",
    platform.python_version()
)


# =========================================================
# CARGAR MODELO
# =========================================================

model = load_model("keras_model.h5")

data = np.ndarray(
    shape=(1, 224, 224, 3),
    dtype=np.float32
)


# =========================================================
# TITULO
# =========================================================

st.title("💜 Reconocimiento de Imágenes")


st.write(
    "Utiliza la cámara para capturar una imagen "
    "y dejar que el modelo identifique el resultado."
)


# =========================================================
# IMAGEN ANIMETECH
# =========================================================

image = Image.open("animetech.avif")

st.image(
    image,
    width=350
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.subheader(
        "✨ Modelo de reconocimiento"
    )

    st.write(
        "Usando un modelo entrenado en "
        "Teachable Machine puedes usarlo "
        "en esta app para identificar."
    )


# =========================================================
# CAMARA
# =========================================================

img_file_buffer = st.camera_input(
    "📸 Toma una Foto"
)


# =========================================================
# PROCESAMIENTO
# =========================================================

if img_file_buffer is not None:

    # To read image file buffer with OpenCV:
    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    # To read the image as a PIL Image:
    img = Image.open(
        img_file_buffer
    )

    newsize = (224, 224)

    img = img.resize(
        newsize
    )

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (
        img_array.astype(np.float32) / 127.0
    ) - 1

    # Load the image into the array
    data[0] = normalized_image_array


    # =====================================================
    # INFERENCE
    # =====================================================

    prediction = model.predict(data)

    print(prediction)


    # =====================================================
    # RESULTADOS
    # =====================================================

    if prediction[0][0] > 0.5:

        st.header(
            "💜 hola salito"
        )


    if prediction[0][1] > 0.5:

        st.header(
            "🧡 hola novio"
        )


    # if prediction[0][2] > 0.5:
    #     st.header(
    #         'Derecha, con Probabilidad: '
    #         + str(prediction[0][2])
    #     )
