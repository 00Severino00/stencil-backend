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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(15)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Limitar resolución para salvar la RAM de Render
    alto, ancho = img.shape[:2]
    max_dim = 1024
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Convertir a escala de grises y suavizado bilateral (mantiene bordes vivos, destruye ruido)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bi = cv2.bilateralFilter(gray, 7, 50, 50)

    # 3. FILTRO CLAHE: Ecualización de contraste local para borrar artefactos de compresión y capturas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_equalizada = clahe.apply(bi)

    # 4. Cálculo dinámico del grosor y sensibilidad de línea
    # Mapeamos la tolerancia del slider (5 a 45) a los parámetros de binarización
    param_c = int((tolerance - 25) / 2)

    # Generamos la base del calco usando un bloque dinámico
    block_size = int(tolerance)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, min(49, block_size))

    # Umbral adaptativo corregido con factor de descarte C para eliminar sombreados indeseados
    stencil = cv2.adaptiveThreshold(
        gray_equalizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, param_c + 5
    )

    # 5. Operación morfológica de limpieza para remover puntos negros aislados (ruido)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(stencil, cv2.MORPH_OPEN, kernel)

    _, encoded_img = cv2.imencode('.png', cleaned)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Desvanecer los serruchos y pixelados usando desenfoque medio y umbralizado directo
    blur = cv2.medianBlur(img, 3)
    _, hd_final = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    # Suavizado de contorno final (antialiasing analógico)
    hd_final = cv2.GaussianBlur(hd_final, (3, 3), 0)
    _, hd_final = cv2.threshold(hd_final, 127, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', hd_final)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
