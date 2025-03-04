import streamlit as st
from streamlit_option_menu import option_menu
from translations import translations

# Default language
if "language" not in st.session_state:
    st.session_state.language = "en"

# Language Selection
selected_language = option_menu(
    menu_title=None,
    options=["English", "हिन्दी", "मराठी"],
    icons=["globe", "globe", "globe"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Map selection to language code
language_map = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
st.session_state.language = language_map[selected_language]

# Get translated text
t = translations[st.session_state.language]

# UI Elements
st.title(t["title"])
st.write(t["description"])

uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if st.button(t["detect_button"]):
    st.write("Processing Image...")  # Call your model here
