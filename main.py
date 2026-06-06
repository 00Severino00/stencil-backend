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
async def generate_stencil(file: UploadFile = File(...), tolerance: int = Form(15), thickness: int = Form(2)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Optimización de resolución inteligente
    alto, ancho = img.shape[:2]
    max_dim = 1400
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)
        alto, ancho = img.shape[:2]

    # 2. Conversión y filtrado sutil de ruido de capturas de pantalla
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Umbralizado Adaptativo regulado por el slider de Tolerancia
    # Mapeamos inversamente para que a mayor tolerancia en la UI, capture líneas más tenues.
    c_param = 25 - int(tolerance)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, c_param
    )

    # 4. Motor de curvas suaves (Estilógrafo)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    
    # Creamos un lienzo 100% limpio (adiós a manchas grises o fondos oscuros de interfaz)
    canvas = np.ones((alto, ancho), dtype=np.uint8) * 255

    # Dibujamos cada contorno con Anti-Aliasing puro
    grosor = max(1, int(thickness))
    for c in contours:
        # Filtro de escala para ignorar imperfecciones o polvillo del fondo
        if cv2.contourArea(c) > 2 or cv2.arcLength(c, True) > 10:
            cv2.drawContours(canvas, [c], -1, (0, 0, 0), thickness=grosor, lineType=cv2.LINE_AA)

    _, encoded_img = cv2.imencode('.png', canvas)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
