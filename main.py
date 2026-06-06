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

    # 1. Redimensión ligera obligatoria para proteger la RAM de Render
    alto, ancho = img.shape[:2]
    max_dim = 800
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        img = cv2.resize(img, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)

    # 2. Convertir a grises y suavizar para remover el ruido digital
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. FILTRO DE ESTILÓGRAFO (Scharr Gradient)
    # Detecta los bordes verdaderos de la imagen (siluetas de ojos, pelo, etc.) e ignora las sombras masivas
    grad_x = cv2.Scharr(blur, cv2.CV_16S, 1, 0)
    grad_y = cv2.Scharr(blur, cv2.CV_16S, 0, 1)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combinamos ambos ejes para tener el mapa de líneas completo
    contornos = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 4. Invertir y aplicar el Grosor del Pincel usando el slider
    # El slider controla el contraste del trazo
    grosor_ajuste = max(5, min(250, int(tolerance) * 6))
    _, thresh = cv2.threshold(contornos, grosor_ajuste, 255, cv2.THRESH_BINARY_INV)

    # 5. Limpieza final anti-aliasing ligera sin consumo de memoria
    resultado = cv2.medianBlur(thresh, 3)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Refinado rápido: adelgaza imperfecciones remanentes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    resultado_hd = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    _, encoded_img = cv2.imencode('.png', resultado_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
