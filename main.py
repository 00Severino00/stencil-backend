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

    # Redimensión de seguridad para control estricto de memoria en Render
    alto, ancho = img.shape[:2]
    max_dim = 1000
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 1. Pasar a escala de grises y suavizar texturas de fondo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Detección autónoma del tipo de imagen (Fondo Oscuro vs Fondo Claro)
    muestra_fondo = gray_blur[0:15, 0:15]
    es_fondo_oscuro = np.mean(muestra_fondo) < 110

    if es_fondo_oscuro:
        # Modo Invertido (Para imágenes tipo Medusa con líneas claras)
        gray_blur = cv2.bitwise_not(gray_blur)

    # 3. Umbralizado de Otsu + Ajuste manual fino (Evita el granulado sucio)
    # Calculamos el umbral óptimo del contraste automáticamente
    umbral_optimo, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Modificamos el umbral usando la tolerancia del slider para limpiar imperfecciones
    ajuste_umbral = umbral_optimo + (tolerance - 15) * 1.5
    ajuste_umbral = max(10, min(245, ajuste_umbral))

    _, thresh = cv2.threshold(gray_blur, ajuste_umbral, 255, cv2.THRESH_BINARY)

    # 4. Suavizado morfológico final para compactar las líneas rotas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    resultado = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Refinado estricto: Elimina los bordes pixelados mediante interpolación bilineal
    alto, ancho = img.shape[:2]
    img_hd = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_LINEAR)
    
    blur = cv2.GaussianBlur(img_hd, (3, 3), 0)
    _, final_hd = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
