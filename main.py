from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import io
import logging
import argparse
import os
import openai
from models_vit import RETFound_mae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Labels for the RFMiD dataset in order corresponding to model outputs
disease_labels = [
    "NORMAL", "CNV", "DME", "DRUSEN", "CSCR", "MH", "RP", "TSLN", "ODC",
    "AION", "ARMD", "BRVO", "CRVO", "CRS", "CSR", "EDN", "ERM", "HTN",
    "MS", "MYA", "ODP", "PRH", "ROP", "STR", "TMH", "VH", "PT"
]

# Initialize FastAPI
app = FastAPI(title="RETFound Disease Detection API")

# Build model
logger.info("Creating model...")
model = RETFound_mae(
    img_size=224,
    num_classes=27,
    drop_path_rate=0.2,
    global_pool=True,
)

checkpoint_path = "models/checkpoint-best.pth"
logger.info("Loading checkpoint from %s", checkpoint_path)
try:
    # Allow loading of argparse.Namespace objects under torch>=2.6
    if hasattr(torch.serialization, "_allowed_globals"):
        torch.serialization._allowed_globals.add("argparse.Namespace")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "Standard checkpoint load failed (%s). Falling back to weights_only.",
            e,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.info("Missing keys when loading state dict: %s", missing_keys)
    if unexpected_keys:
        logger.info("Unexpected keys when loading state dict: %s", unexpected_keys)
    model.eval()
    load_error = None
except Exception as e:  # pylint: disable=broad-except
    logger.exception("Model initialization failed")
    model = None
    load_error = str(e)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": f"Model not loaded: {load_error}"})
    try:
        image_bytes = await file.read()
        logger.info("Received prediction request: %s (%d bytes)", file.filename, len(image_bytes))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        logger.info("Tensor shape after transforms: %s", tuple(input_tensor.shape))

        with torch.no_grad():
            try:
                outputs = model(input_tensor)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception("Model inference failed")
                return JSONResponse(status_code=500, content={"error": str(e)})
            probs = torch.sigmoid(outputs).squeeze().tolist()

        # Pair each probability with its corresponding disease label
        predictions = [
            {"disease": disease_labels[i], "probability": float(prob)}
            for i, prob in enumerate(probs)
        ]

        # Sort predictions by probability in descending order
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        # Only keep the top 5 predictions
        top_predictions = predictions[:5]

        # Build a prompt for GPT to summarise these diseases
        diseases = ", ".join(p["disease"] for p in top_predictions)
        prompt = (
            "Explain in simple terms what the following retinal diseases are. "
            "For each disease, include a short explanation, common symptoms, and "
            f"potential treatments: {diseases}."
        )

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            summary = completion["choices"][0]["message"]["content"].strip()
        except Exception as e:  # pylint: disable=broad-except
            logger.exception("OpenAI API call failed")
            summary = f"Failed to generate explanation: {e}"

        return JSONResponse(content={"predictions": top_predictions, "summary": summary})
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
