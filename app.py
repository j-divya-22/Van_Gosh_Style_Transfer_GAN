import base64
import hashlib
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FINE_TUNED_MODEL_PATH = (
    BASE_DIR / "models" / "fine_tuned" / "vangogh_generator.pth"
)
ORIGINAL_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "original_pretrained"
    / "verified_vangogh_generator.pth"
)
BACKGROUND_IMAGE = base64.b64encode(
    (BASE_DIR / "assets" / "starry-night-background.png").read_bytes()
).decode("ascii")

st.set_page_config(
    page_title="Van Gogh AI Studio",
    page_icon="🎨",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --text-color: #f3f5ff;
        --primary-color: #f5d875;
        --secondary-background-color: #102c70;
        --starry-navy: #071536;
        --deep-blue: #102c70;
        --cobalt: #2857b8;
        --sky-blue: #78a8d7;
        --moonlight: #f5d875;
    }

    .stApp {
        background-color: #071536;
        background-image:
            linear-gradient(180deg, rgba(5, 15, 42, 0.38), rgba(5, 15, 42, 0.56)),
            url("__BACKGROUND_IMAGE__");
        background-position: center center;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #f3f5ff;
    }

    .stApp p,
    .stApp label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCaptionContainer"] {
        color: #f3f5ff !important;
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        inset: -55%;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse at 22% 44%, rgba(164, 202, 231, 0.17) 0 3%, transparent 17%),
            radial-gradient(ellipse at 38% 36%, rgba(101, 155, 213, 0.19) 0 4%, transparent 20%),
            radial-gradient(ellipse at 58% 48%, rgba(146, 189, 221, 0.16) 0 3%, transparent 17%),
            radial-gradient(ellipse at 76% 38%, rgba(88, 140, 204, 0.2) 0 5%, transparent 22%);
        filter: blur(24px);
        opacity: 0.72;
        animation: cloud-swirl 72s linear infinite;
        will-change: transform;
    }

    .stApp::after {
        inset: -38%;
        background:
            radial-gradient(ellipse at 30% 60%, rgba(164, 202, 231, 0.13) 0 3%, transparent 18%),
            radial-gradient(ellipse at 50% 30%, rgba(82, 133, 195, 0.16) 0 4%, transparent 20%),
            radial-gradient(ellipse at 77% 59%, rgba(140, 181, 216, 0.12) 0 3%, transparent 17%);
        filter: blur(32px);
        opacity: 0.58;
        animation: cloud-drift 94s linear infinite reverse;
    }

    .stApp > header,
    .stApp > div {
        position: relative;
        z-index: 1;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 21, 54, 0.42);
    }

    [data-testid="stFileUploader"] section {
        background: rgba(15, 36, 82, 0.68);
        border: 1px solid rgba(120, 168, 215, 0.55);
        border-radius: 16px;
    }

    [data-testid="stFileUploader"] button {
        color: #f3f5ff;
        background: #17336d;
        border: 1px solid rgba(120, 168, 215, 0.65);
    }

    [data-testid="stImage"] img {
        border: 1px solid rgba(245, 216, 117, 0.42);
        border-radius: 12px;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1.25rem;
    }

    [data-testid="stButton"] button[kind="primary"],
    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid #f5d875;
        color: #071536;
        background: linear-gradient(135deg, #f8e394, #e8bd4f);
        font-weight: 700;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    [data-testid="stButton"] button[kind="primary"]:hover,
    .stDownloadButton > button:hover {
        color: #071536;
        border-color: #fff0a8;
        box-shadow: 0 0 22px rgba(245, 216, 117, 0.38);
        transform: translateY(-2px);
    }

    @keyframes cloud-swirl {
        from { transform: rotate(0deg) scale(1.02); }
        to { transform: rotate(360deg) scale(1.02); }
    }

    @keyframes cloud-drift {
        from { transform: rotate(360deg) translate3d(-1.5%, 1%, 0); }
        to { transform: rotate(0deg) translate3d(1.5%, -1%, 0); }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp::before,
        .stApp::after {
            animation: none;
        }

        [data-testid="stButton"] button[kind="primary"],
        .stDownloadButton > button {
            transition: none;
        }
    }
    </style>
    """.replace(
        "__BACKGROUND_IMAGE__",
        f"data:image/png;base64,{BACKGROUND_IMAGE}",
    ),
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(model_path: str):
    from model import load_generator

    return load_generator(Path(model_path))


st.title("🎨 Van Gogh AI Studio")
st.write("Transform your photographs into Van Gogh-inspired paintings.")

st.subheader("Upload your photograph")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    uploaded_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(uploaded_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(uploaded_bytes)) as uploaded_image:
            original = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not open the uploaded image: {error}")
    else:
        st.image(
            original,
            caption="Uploaded image preview",
            width="stretch",
        )

        model_buttons = st.columns(2)
        with model_buttons[0]:
            fine_tuned_clicked = st.button(
                "✨ Generate with My Fine-Tuned Model",
                type="primary",
                width="stretch",
            )
        with model_buttons[1]:
            original_clicked = st.button(
                "🎨 Generate with Original Pretrained Model",
                type="primary",
                width="stretch",
            )

        selected_model = None
        if fine_tuned_clicked:
            selected_model = (
                "My Fine-Tuned Model",
                FINE_TUNED_MODEL_PATH,
                "vangogh_fine_tuned.png",
            )
        elif original_clicked:
            selected_model = (
                "Original Pretrained Model",
                ORIGINAL_MODEL_PATH,
                "vangogh_original_pretrained.png",
            )

        if selected_model is not None:
            model_name, model_path, download_name = selected_model
            st.session_state.pop("generation_result", None)
            try:
                with st.spinner(f"Generating with the {model_name}..."):
                    model, device = get_model(str(model_path))
                    from model import convert_image

                    generated = convert_image(original, model, device)

                    buffer = io.BytesIO()
                    generated.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    generated.save(OUTPUT_DIR / download_name)

                st.session_state["generation_result"] = {
                    "upload_id": upload_id,
                    "model_name": model_name,
                    "download_name": download_name,
                    "image_bytes": image_bytes,
                }
                st.success(f"{model_name} generated the painting successfully.")
            except FileNotFoundError as error:
                st.error(f"Could not find the {model_name} weights: {error}")
            except RuntimeError as error:
                st.error(
                    f"The {model_name} could not load or run the model. "
                    f"Check that its checkpoint matches the generator architecture. {error}"
                )
            except Exception as error:
                st.error(f"The {model_name} could not generate the image: {error}")

        result = st.session_state.get("generation_result")
        if result is not None and result["upload_id"] == upload_id:
            st.subheader(f"Comparison — {result['model_name']}")
            original_column, generated_column = st.columns(2)
            with original_column:
                st.image(
                    original,
                    caption="Original photograph",
                    width="stretch",
                )
            with generated_column:
                generated_image = Image.open(
                    io.BytesIO(result["image_bytes"])
                )
                st.image(
                    generated_image,
                    caption=result["model_name"],
                    width="stretch",
                )
                st.download_button(
                    label="⬇️ Download painting",
                    data=result["image_bytes"],
                    file_name=result["download_name"],
                    mime="image/png",
                    width="stretch",
                )
else:
    st.info("Upload a photograph to preview it and choose a model.")
