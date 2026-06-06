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

    # 1. Convertir a escala de grises directa (ahorra memoria)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Filtro de caja ligero para unificar los bordes pixelados sin consumir RAM
    # Esto elimina el ruido gris de las capturas de pantalla de inmediato
    suave = cv2.blur(gray, (3, 3))
    
    # 3. Umbralizado de precisión manual basado en el slider
    # Mapeamos el slider para controlar el contraste de la tinta directamente
    # Un valor más bajo limpia el fondo; un valor más alto rescata líneas tenues
    limite = max(50, min(240, 255 - (int(tolerance) * 3)))
    
    # Creamos un line-art puro: lo que es oscuro se vuelve negro sólido, lo claro blanco puro
    _, stencil_puro = cv2.threshold(suave, limite, 255, cv2.THRESH_BINARY)

    # 4. Suavizado sub-píxel ultra-ligero para eliminar el "diente de sierra"
    # Esto le da el acabado fluido del estilógrafo sin sobrecargar el servidor
    resultado = cv2.GaussianBlur(stencil_puro, (3, 3), 0)
    _, resultado = cv2.threshold(resultado, 200, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', resultado)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

@app.post("/render-hd/")
async def render_hd(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Render HD optimizado para el plan gratuito de Render (No genera caídas de memoria)
    # Dilatamos un píxel para rellenar microporos en las curvas y luego suavizamos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_consolidada = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    final_hd = cv2.GaussianBlur(img_consolidada, (3, 3), 0)
    _, final_hd = cv2.threshold(final_hd, 220, 255, cv2.THRESH_BINARY)

    _, encoded_img = cv2.imencode('.png', final_hd)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
