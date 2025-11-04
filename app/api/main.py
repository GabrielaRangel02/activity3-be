python
import io, os, json, time
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn
import requests

IMG_SIZE = (224, 224)
MODEL_PATH = "app/model/model.h5"
LABELS_PATH = "app/model/labels.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Mask Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES: List[str] = json.load(f)

class PredictResponse(BaseModel):
    klass: str
    prob: float
    explanation: str
    recommendations: List[str]
    latency_ms: int
    ts: float

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def call_openai_explainer(klass: str, prob: float):
    if not OPENAI_API_KEY:
        if klass == "correcta":
            return "La mascarilla parece colocada adecuadamente (nariz y boca cubiertas).", [
                "Mantener cubierta la nariz y boca", "Ajustar el puente nasal para buen sello"
            ]
        elif klass == "incorrecta":
            return "La mascarilla parece mal ajustada o sin cubrir completamente nariz/boca.", [
                "Ajustar el puente nasal", "Subir la mascarilla para cubrir nariz y boca"
            ]
        else:
            return "No se detecta uso de mascarilla.", [
                "Colocarse una mascarilla homologada", "Ajustarla correctamente cubriendo nariz y boca"
            ]

    prompt = (
        f"Eres un asistente breve y claro de bioseguridad. "
        f"Clase detectada: '{klass}' con probabilidad {prob:.2f}. "
        f"Redacta 1 explicación corta (1 frase) y 2 recomendaciones puntuales. "
        f"Devuelve en JSON: {{explanation, recommendations}}."
    )
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Responde siempre en JSON válido."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        explanation = data.get("explanation", "") or f"Resultado: {klass}."
        recs = data.get("recommendations", [])[:2] or ["Verificar sellado nasal", "Cubrir nariz y boca correctamente"]
        return explanation, recs
    except Exception:
        if klass == "correcta":
            return "La mascarilla parece colocada adecuadamente (nariz y boca cubiertas).", [
                "Mantener cubierta la nariz y boca", "Ajustar el puente nasal para buen sello"
            ]
        elif klass == "incorrecta":
            return "La mascarilla parece mal ajustada o sin cubrir completamente nariz/boca.", [
                "Ajustar el puente nasal", "Subir la mascarilla para cubrir nariz y boca"
            ]
        else:
            return "No se detecta uso de mascarilla.", [
                "Colocarse una mascarilla homologada", "Ajustarla correctamente cubriendo nariz y boca"
            ]

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo inválido. Sube una imagen.")

    start = time.time()
    x = preprocess(pil)
    logits = model.predict(x, verbose=0)[0]
    probs = logits if np.isclose(np.sum(logits), 1.0, atol=1e-3) else softmax(logits)
    idx = int(np.argmax(probs))
    klass = CLASS_NAMES[idx]
    prob  = float(probs[idx])

    explanation, recommendations = call_openai_explainer(klass, prob)
    latency_ms = int((time.time() - start) * 1000)
    return PredictResponse(
        klass=klass, prob=prob, explanation=explanation,
        recommendations=recommendations, latency_ms=latency_ms, ts=time.time()
    )

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8080, reload=False)