from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-stencil/")
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(10)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Pasar a escala de grises de alta precisión
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Duplicar el tamaño con interpolación cúbica para crear píxeles intermedios suavizados
    alto, ancho = gray.shape[:2]
    gray_scaled = cv2.resize(gray, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    
    # 3. Extraer bordes usando gradiente morfológico (genera el contorno del estilógrafo)
    # Ajustamos el tamaño del pincel según el slider de tolerancia de forma sutil
    grosor_pincel = 3 if tolerance < 10 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grosor_pincel, grosor_pincel))
    gradiente = cv2.morphologyEx(gray_scaled, cv2.MORPH_GRADIENT, kernel)
    
    # 4. Invertir y aplicar umbral con desenfoque sub-píxel para simular la tinta del estilógrafo
    # Mapeamos la tolerancia para ajustar qué tan "fuerte" o sensible es la presión del trazo
    umbral_dinamico = max(5, int(tolerance * 4))
    _, fondo_blanco = cv2.threshold(gradiente, umbral_dinamico, 255, cv2.THRESH_BINARY_INV)
    
    # 5. El toque de Procreate: Un suavizado gaussiano leve que elimina los dientes de sierra
    line_art_pulido = cv2.GaussianBlur(fondo_blanco, (3, 3), 0)

    _, encoded_img = cv2.imencode('.png', line_art_pulido)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # El Render HD refina y consolida el suavizado del estilógrafo
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, final_hd = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
