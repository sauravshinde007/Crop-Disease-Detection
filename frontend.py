import streamlit as st
from streamlit_option_menu import option_menu
from translations import translations
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import json
import tempfile

# Define CBAM Layer
class CBAMLayer(torch.nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels, bias=False),
            torch.nn.Sigmoid()
        )
        self.spatial_conv = torch.nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_bn = torch.nn.BatchNorm2d(1)
        self.spatial_sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        channel_att = self.fc(avg_out) + self.fc(max_out)
        channel_att = channel_att.view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        spatial_att = self.spatial_bn(spatial_att)
        spatial_att = self.spatial_sigmoid(spatial_att)
        return x * spatial_att
    
# Define EfficientNet with CBAM
class EfficientNetCBAM(torch.nn.Module):
    def __init__(self, version='b0', num_classes=10):
        super(EfficientNetCBAM, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(f'efficientnet-{version}')
        
        # Adding CBAM to specific layers
        self.cbam1 = CBAMLayer(in_channels=24)  # Example early layer
        self.cbam2 = CBAMLayer(in_channels=112) # Deeper layer
        
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = torch.nn.Linear(num_ftrs, num_classes)

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

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Cached model loader
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetCBAM(version='b3', num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load('efficientnet_cbam_model.pth', map_location=device))
    model.eval()
    return model, device

# Define transformations
def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Modified predict function
def predict(image, crop_name, model, device, transform):
    try:
        image = Image.open(image).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
        
        crop_classes = [cls for cls in class_names if cls.startswith(crop_name)]
        crop_indices = [class_names.index(cls) for cls in crop_classes]
        
        crop_output = output[:, crop_indices]
        predicted_idx = torch.argmax(crop_output, dim=1).item()
        predicted_disease = crop_classes[predicted_idx]
        
        confidence = torch.nn.functional.softmax(output, dim=1)[0, crop_indices[predicted_idx]].item()
        disease_name = predicted_disease.split("_", 1)[1]
        
        return disease_name, confidence
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

# Streamlit UI
def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
            .main-title {
                font-size: 50px !important;
                font-weight: bold;
                color: #FFFFFF;
                text-align: center;
                padding: 20px;
            }
            .sub-title {
                font-size: 30px !important;
                color: #148F77;
                text-align: center;
                padding: 10px;
            }
            .section-title {
                font-size: 25px !important;
                color: #1A5276;
                padding: 10px 0px;
            }
            .info-text {
                font-size: 18px !important;
                color: #2C3E50;
                padding: 5px 0px;
            }
            .stButton>button {
                background-color: #1ABC9C !important;
                color: white !important;
                font-size: 18px !important;
                padding: 10px 24px;
                border-radius: 8px;
            }
            .stFileUploader>div>div>div>div {
                color: #1A5276;
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Language Selection
    if "language" not in st.session_state:
        st.session_state.language = "en"

    selected_language = option_menu(
        menu_title=None,
        options=["English", "हिन्दी", "मराठी"],
        icons=["globe", "globe", "globe"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    language_map = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
    st.session_state.language = language_map[selected_language]
    t = translations[st.session_state.language]

    # Main Title and Description
    st.markdown(f'<div class="main-title">{t["main_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{t["sub_title"]}</div>', unsafe_allow_html=True)
    
    # About Section
    with st.expander(t["about_section_title"]):
        st.markdown(f'<div class="info-text">{t["about_text"]}</div>', unsafe_allow_html=True)
    
    # How It Works Section
    with st.expander(t["how_it_works_title"]):
        st.markdown(f"""
        <div class="info-text">
            <ol>
                <li>{t["step1"]}</li>
                <li>{t["step2"]}</li>
                <li>{t["step3"]}</li>
                <li>{t["step4"]}</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Crop Selection
    st.markdown(f'<div class="section-title">{t["crop_selection_title"]}</div>', unsafe_allow_html=True)
    crop_options = list(sorted(set([name.split('_')[0] for name in class_names])))
    selected_crop = st.selectbox(t["select_crop"], options=crop_options)
    
    # Image Upload Section
    st.markdown(f'<div class="section-title">{t["upload_section_title"]}</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])
    
    # Detection Section
    if uploaded_file and selected_crop:
        st.markdown(f'<div class="section-title">{t["detection_section_title"]}</div>', unsafe_allow_html=True)
        if st.button(t["detect_button"]):
            try:
                # Load model and prepare image
                model, device = load_model()
                transform = get_transform()
                
                # Process image
                img = Image.open(uploaded_file).convert("RGB")
                
                # Make prediction
                disease_name, confidence = predict(uploaded_file, selected_crop, model, device, transform)
                
                # Display results
                st.image(img, caption=t["uploaded_image_caption"], use_column_width=True)
                st.success(f"""
                **{t['prediction_result']}**
                - {t['crop_label']}: {selected_crop}
                - {t['disease_label']}: {disease_name}
                - {t['confidence_label']}: {confidence*100:.1f}%
                """)
                
            except Exception as e:
                st.error(f"{t['error_message']}: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(f'<div class="info-text" style="text-align:center;">{t["footer_text"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()