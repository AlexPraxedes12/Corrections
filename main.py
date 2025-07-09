from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import io
import logging
import argparse
from models_vit import RETFound_mae

logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI(title="RETFound Disease Detection API")

# Build model
model = RETFound_mae(
    img_size=224,
    num_classes=27,
    drop_path_rate=0.2,
    global_pool=True,
)

checkpoint_path = "models/checkpoint-best.pth"
try:
    # Allow loading of argparse.Namespace objects under torch>=2.6
    if hasattr(torch.serialization, "_allowed_globals"):
        torch.serialization._allowed_globals.add(argparse.Namespace)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Standard checkpoint load failed (%s). Retrying with weights_only.", e
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    load_error = None
except Exception as e:  # pylint: disable=broad-except
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
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().tolist()

        return JSONResponse(content={"probabilities": probs})
    except Exception as e:  # pylint: disable=broad-except
        return JSONResponse(status_code=500, content={"error": str(e)})
