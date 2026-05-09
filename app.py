# Program title: Storytelling App

import streamlit as st
from transformers import pipeline
import tempfile
import os
import re


# -----------------------------
# Cache models
# -----------------------------
@st.cache_resource
def load_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_model():
    # Lightweight model, more suitable for Streamlit Cloud
    return pipeline(
        "text2text-generation",
        model="mehuldamani/story-gen_llama-sft-full"
    )


@st.cache_resource
def load_audio_model():
    return pipeline(
        "text-to-audio",
        model="facebook/mms-tts-eng"
    )


# -----------------------------
# Safety filter
# -----------------------------
FORBIDDEN_WORDS = [
    "murder", "kill", "killed", "killer", "blood", "bloody",
    "gun", "knife", "weapon", "gangster", "crime", "criminal",
    "death", "dead", "drug", "alcohol", "horror", "scary",
    "terror", "violence", "violent", "suicide"
]


def is_safe(text):
    text = text.lower()

    for word in FORBIDDEN_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text):
            return False

    return True


# -----------------------------
# Image to text
# -----------------------------
def img2text(image_path):
    image_to_text_model = load_img2text_model()
    result = image_to_text_model(image_path)
    text = result[0]["generated_text"]
    return text


# -----------------------------
# Generate child-safe story
# -----------------------------
def generate_child_story(scenario, max_attempts=5):
    story_model = load_story_model()

    for attempt in range(max_attempts):
        prompt = f"""
        Write a short, warm, child-friendly bedtime story for children aged 3 to 11.

        Image description: {scenario}

        Requirements:
        - The story must be safe for young children.
        - No violence.
        - No murder.
        - No gangster.
        - No weapons.
        - No scary horror content.
        - No death.
        - No adult content.
        """

        result = story_model(
            prompt,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        story = result[0]["generated_text"].strip()

        if is_safe(story):
            return story

    return None


# -----------------------------
# Main app
# -----------------------------
st.set_page_config(
    page_title="Your Image to Audio Story",
    page_icon="🦜"
)

st.header("Turn Your Image into an Audio Story")

uploaded_file = st.file_uploader(
    "Select an Image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Stage 1: Image to Text
        with st.spinner("Processing image..."):
            scenario = img2text(image_path)

        st.write("**Scenario:**")
        st.write(scenario)

        # Stage 2: Text to Story
        with st.spinner("Generating a child-safe story..."):
            story = generate_child_story(scenario, max_attempts=5)

        if story is None:
            st.warning(
                "The app could not generate a safe story this time. "
                "Please try another image or click rerun."
            )
            st.stop()

        st.write("**Story:**")
        st.write(story)

        # Stage 3: Story to Audio
        with st.spinner("Generating audio..."):
            audio_pipe = load_audio_model()
            audio_data = audio_pipe(story)

        st.audio(
            audio_data["audio"],
            sample_rate=audio_data["sampling_rate"]
        )

        # Remove temporary image file
        os.remove(image_path)

    except Exception as e:
        st.error("Something went wrong.")
        st.exception(e)
