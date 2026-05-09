
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
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
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
    "murder", "kill", "killed", "killer",
    "blood", "bloody",
    "gun", "guns",
    "knife", "knives",
    "weapon", "weapons",
    "gangster", "gang",
    "crime", "criminal", "robber",
    "death", "dead", "die", "died",
    "drug", "drugs", "alcohol",
    "horror", "scary", "terror",
    "violence", "violent",
    "suicide", "abuse",
    "fight", "fighting",
    "war", "monster"
]


def keyword_safe(text):
    text_lower = text.lower()

    for word in FORBIDDEN_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text_lower):
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
# Build prompt
# -----------------------------
def build_story_prompt(scenario, attempt):
    prompt = f"""
You are writing a story for children aged 3 to 11.

Image description:
{scenario}

Task:
Write a short, gentle, child-friendly bedtime story inspired by the image.

Very important rules:
- The story must be safe for young children.
- Do not include violence.
- Do not include murder.
- Do not include blood.
- Do not include weapons.
- Do not include gangsters.
- Do not include crime.
- Do not include death.
- Do not include horror.
- Do not include scary scenes.
- Do not include adult content.
- Do not include dangerous behavior.
- If the image description contains unsafe elements, ignore those elements completely.
- Focus only on safe elements such as colors, animals, nature, friendship, family, school, toys, kindness, curiosity, imagination, and learning.
- Use simple vocabulary.
- Make the story warm, positive, educational, and comforting.
- The story should have a happy ending.
- Around 100 to 150 words.

Story structure:
1. Introduce a friendly character.
2. Describe a peaceful place.
3. Let the character discover something interesting.
4. Add a simple lesson about kindness, curiosity, or friendship.
5. End happily.

Only output the story. Do not explain.

Attempt: {attempt}
"""
    return prompt


# -----------------------------
# Generate child-safe story
# -----------------------------
def generate_child_story(scenario, max_attempts=5):
    story_model = load_story_model()

    for attempt in range(1, max_attempts + 1):
        prompt = build_story_prompt(scenario, attempt)

        result = story_model(
            prompt,
            max_new_tokens=220,
            do_sample=False,
            num_beams=5,
            early_stopping=True
        )

        story = result[0]["generated_text"].strip()

        if keyword_safe(story):
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
            story = generate_child_story(scenario, max_attempts=5)

        if story is None:
            st.warning(
                "The app could not generate a safe story this time. "
                "Please try another image."
            )
            os.remove(image_path)
            st.stop()

        st.write("**Story:**")
        st.write(story)

        # Final safety check before audio
        if not keyword_safe(story):
            st.warning(
                "The generated story may not be suitable for children, "
                "so audio was not generated."
            )
            os.remove(image_path)
            st.stop()

        # Stage 3: Story to Audio
        with st.spinner("Generating audio..."):
            audio_pipe = load_audio_model()
            audio_data = audio_pipe(story)

        st.audio(
            audio_data["audio"],
            sample_rate=audio_data["sampling_rate"]
        )

        os.remove(image_path)

    except Exception as e:
        st.error("Something went wrong.")
        st.exception(e)
