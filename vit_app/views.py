import torch
from PIL import Image
import os, json
from django.shortcuts import render
from django.core.files.storage import default_storage
from torchvision import transforms
from transformers import ViTForImageClassification
import gdown
import zipfile

def download_model_if_needed():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading model from Google Drive...")

        file_id = "1Z7q9uv0hWvTOhZCir6MLJoyTlXXgYtC7"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "vit_cataract_model.zip"

        gdown.download(url, output, quiet=False)

        # Extract isi zip tanpa folder root
        with zipfile.ZipFile(output, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Hilangkan folder root (misal: 'vit_cataract_model/')
                stripped = os.path.relpath(member, start=zip_ref.namelist()[0].split('/')[0])
                target_path = os.path.join(MODEL_DIR, stripped)

                if member.endswith('/'):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        os.remove(output)
        print("Model downloaded and extracted to:", MODEL_DIR)

# Load model sekali saat server start
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'vit_cataract_model')
download_model_if_needed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(MODEL_DIR, use_safetensors=True)
model.to(device)
model.eval()

# Class mapping
with open(os.path.join(MODEL_DIR, 'class_to_idx.json'), 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image).logits
        _, pred = torch.max(logits, 1)
    return idx_to_class[pred.item()]

def index(request):
    prediction = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_path = default_storage.save('uploads/' + image_file.name, image_file)
        full_path = os.path.join(default_storage.location, image_path)
        prediction = predict_image(full_path)
        image_url = default_storage.url(image_path)

    return render(request, 'vit_app/index.html', {
        'prediction': prediction,
        'image_url': image_url
    })
