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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(25)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Control estricto de resolución (Cero caídas de RAM)
    alto, ancho = img.shape[:2]
    max_dim = 1000
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Conversión a escala de grises y eliminación de ruido de compresión
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Suavizado bilateral para conservar bordes artísticos vivos e ignorar texturas suaves
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Detector de Bordes Canny con umbral dinámico basado en el Slider
    # Mapeamos la tolerancia para ajustar la sensibilidad del trazo técnico
    thresh_high = max(50, min(250, int(tolerance * 5)))
    thresh_low = int(thresh_high / 2)
    
    edges = cv2.Canny(smoothed, thresh_low, thresh_high)

    # 5. Invertir para obtener líneas negras sobre fondo blanco (Estilo Calco Tradicional)
    stencil = cv2.bitwise_not(edges)

    # 6. Limpieza morfológica para conectar líneas finas rotas y eliminar imperfecciones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    resultado = cv2.morphologyEx(stencil, cv2.MORPH_CLOSE, kernel)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Suavizado sub-píxel para remover el diente de sierra
    blur = cv2.medianBlur(img, 3)
    _, hd_final = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # Dilatación mínima para darle cuerpo al trazo de la aguja si quedó muy delgado
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    hd_final = cv2.morphologyEx(hd_final, cv2.MORPH_MINIFY if False else cv2.MORPH_AND, kernel)

    _, encoded_img = cv2.imencode('.png', hd_final)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
