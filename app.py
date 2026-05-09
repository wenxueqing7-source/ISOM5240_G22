# Program title: Storytelling App

import streamlit as st
from transformers import pipeline
import tempfile
import os

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
    # Lightweight model for Streamlit Cloud
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
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
    "terror", "violence", "violent", "monster"
]

def is_safe(text):
    text = text.lower()
    return not any(word in text for word in FORBIDDEN_WORDS)


# -----------------------------
# Functions
# -----------------------------
def img2text(image_path):
    image_to_text_model = load_img2text_model()
    text = image_to_text_model(image_path)[0]["generated_text"]
    return text


def generate_child_story(scenario):
    story_model = load_story_model()

    prompt = f"""
    Write a short child-friendly bedtime story for children aged 3 to 11.

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
    - Use simple and warm language.
    - Make the story positive, gentle, and educational.
    - The story should be around 100 words.
    """

    result = story_model(
        prompt,
        max_new_tokens=180,
        do_sample=False
    )

    story = result[0]["generated_text"]

    if is_safe(story):
        return story
    else:
        return """
        Once upon a time, a curious little friend found something special in a bright and peaceful place.
        They looked around with wonder and learned something new about the world.
        With kindness, imagination, and a happy heart, the little friend turned an ordinary day into a lovely adventure.
        Everyone felt safe, warm, and joyful.
        """


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
        # Save uploaded image to a temporary file
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
            story = generate_child_story(scenario)

        st.write("**Story:**")
        st.write(story)

        # Stage 3: Story to Audio
        if is_safe(story):
            with st.spinner("Generating audio..."):
                audio_pipe = load_audio_model()
                audio_data = audio_pipe(story)

            st.audio(
                audio_data["audio"],
                sample_rate=audio_data["sampling_rate"]
            )
        else:
            st.warning("The story may not be suitable for children, so audio was not generated.")

        # Remove temporary file
        os.remove(image_path)

    except Exception as e:
        st.error("Something went wrong.")
        st.exception(e)
