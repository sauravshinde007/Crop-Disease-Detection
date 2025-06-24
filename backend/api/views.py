import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import json
import traceback
from PIL import ExifTags,Image
from datetime import datetime
import requests
import os
from django.conf import settings
from rest_framework.response import Response # type: ignore
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

# CBAM Layer
class CBAMLayer(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = (avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        spatial_att = self.spatial_bn(spatial_att)
        spatial_att = self.spatial_sigmoid(spatial_att)
        return x * spatial_att

# EfficientNet with CBAM
class EfficientNetCBAM(nn.Module):
    def __init__(self, version='b0', num_classes=10):
        super(EfficientNetCBAM, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(f'efficientnet-{version}')
        self.cbam1 = CBAMLayer(in_channels=24)
        self.cbam2 = CBAMLayer(in_channels=112)
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        if x.shape[1] == 24:
            x = self.cbam1(x)
        if x.shape[1] == 112:
            x = self.cbam2(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

# Load Class Names
class_names_path = os.path.join(settings.MEDIA_ROOT, "class_names.json")
with open(class_names_path, "r") as f:
    class_names = json.load(f)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(settings.MEDIA_ROOT, "efficientnet_cbam_model.pth")
model = EfficientNetCBAM(version='b3', num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


API_KEY=settings.GEMINI_API_KEY
def get_image_capture_time(image_file):
    try:
        img = Image.open(image_file)
        exif_data = img._getexif()

        if not exif_data:
            return None

        # Reverse mapping from ExifTags
        exif = {
            ExifTags.TAGS.get(k): v
            for k, v in exif_data.items()
            if k in ExifTags.TAGS
        }

        date_str = exif.get("DateTimeOriginal")  # Most accurate
        if date_str:
            return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")

        # Fallbacks
        date_str = exif.get("DateTime")
        if date_str:
            return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        print("EXIF read error:", e)
    return None

def get_coordinates(state, district, api_key):
    location = f"{district},{state},IN"  # 'IN' is the country code for India
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    response = requests.get(geo_url)

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return data['lat'], data['lon']
    return None, None

def translate_text(text, target_lang="hi", source_lang="auto"):
    """
    Translate text using Lingva Translate API.
    :param text: Text to translate
    :param target_lang: Target language code (e.g., 'hi' for Hindi)
    :param source_lang: Source language code ('auto' will auto-detect)
    :return: Translated text
    """
    try:
        base_url = "https://lingva.ml/api/v1"
        url = f"{base_url}/{source_lang}/{target_lang}/{text}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("translation", text)
        else:
            return text  # fallback to original if error
    except Exception as e:
        print("Translation error:", e)
        return text
    
def get_weather(state, district, timestamp=None):
    if not state or not district:
        return {"error": "State and district are required"}

    api_key = settings.OPENWEATHER_API_KEY
    lat, lon = get_coordinates(state, district, api_key)
    if timestamp:
        # Use timestamp-based historical API
        unix_time = int(timestamp.timestamp())
        print("Unix Time:", unix_time)
        url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={{LAT}}&lon={{LON}}&dt={unix_time}&appid={api_key}&units=metric"
        
        # You’ll need a geocoding step to get LAT/LON from state/district
        # Placeholder - you must implement proper geocoding
        0  # Example: New Delhi
        url = url.format(LAT=lat, LON=lon)
    else:
        # Default current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?q={district}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to fetch weather data"}

    data = response.json()
    weather_info = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"]
    }
    return weather_info


    
def get_remedy_from_gemini(disease, crop, confidence,state, district):
    headers = {"Content-Type": "application/json"}
    weather_data = get_weather(state, district)  

    # ✅ Check if weather data is valid
    if "error" in weather_data:
        weather_text = "Weather data not available."
    else:
        weather_text = (
            f"Current weather in {district}, {state}: {weather_data['temperature']}°C, "
            f"Humidity: {weather_data['humidity']}%, Condition: {weather_data['condition']}."
        )


    payload = {
        "contents": [{"parts": [{"text": f"What are the best remedies for {disease} in {crop} with given confidence score {confidence} located at {state},{district} with weather condition there is {weather_text}in 200 words and dont mention confidence score in response and give in paragraphs include cultural practices also? "}]}],
        "generationConfig": {"maxOutputTokens": 200}  # Limit response length
    }
    
    response = requests.post(f"{GEMINI_URL}?key={API_KEY}", headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    else:
        return "No remedy found."


# Example Usage

# API Endpoint
@api_view(['POST'])
def predict(request):
    image = request.FILES.get('image')
    crop_name = request.POST.get('crop')
    state = request.POST.get("state")  # Get state from request
    district = request.POST.get("district")
    target = request.POST.get("language")
    timestamp = get_image_capture_time(image) 
    print("Timestamp",timestamp) # <-- extract timestamp here
    weather_data = get_weather(state, district, timestamp)
    print("Weather Data:", weather_data)  # <-- check the weather data
    if not image or not crop_name:
        return Response({'error': 'Missing image or crop name'}, status=400)
    
    try:
        img = Image.open(image).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
        
        crop_classes = [cls for cls in class_names if cls.startswith(crop_name)]
        crop_indices = [class_names.index(cls) for cls in crop_classes]
        
        crop_output = output[:, crop_indices]
        predicted_idx = torch.argmax(crop_output, dim=1).item()
        predicted_disease = crop_classes[predicted_idx]
        confidence = F.softmax(output, dim=1)[0, crop_indices[predicted_idx]].item()
        disease_name = predicted_disease.split("_", 1)[1]
        
        remedy = get_remedy_from_gemini(disease_name, crop_name, confidence, state, district)
        translated_disease = translate_text(disease_name, target)
        translated_remedy = translate_text(remedy,target)
        translated_weather = translate_text(f"Temperature: {weather_data['temperature']}°C, "
                                           f"Humidity: {weather_data['humidity']}%, "
                                           f"Condition: {weather_data['condition']}")
        
        

        return Response({
            "crop": crop_name,
            "disease": translated_disease,
            "confidence": confidence,
            "remedy": translated_remedy,
            "weather": weather_data
        })

    
    except Exception as e:
        print("Error in /api/predict/:", str(e))
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)
