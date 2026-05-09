
# Program title: Storytelling App

import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import re


# Function 1: Image to Text
def img2text(image_path):
    if "img_model" not in st.session_state:
        st.session_state.img_model = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )

    result = st.session_state.img_model(image_path)
    return result[0]["generated_text"]


# Function 2: Text to Story
def text2story(scenario):
    if "story_model" not in st.session_state:
        st.session_state.story_model = pipeline(
            "text-generation",
            model="roneneldan/TinyStories-Instruct-33M"
        )

    prompt = f"""
Write a short story for a young child.

The story is inspired by this picture: {scenario}

Rules:
- Safe for children aged 3 to 11.
- No violence, no death, no weapons, no scary content.
- Use simple words.
- Make it warm, happy, and positive.
- About friendship, kindness, curiosity, or imagination.

Story:
"""

    result = st.session_state.story_model(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        return_full_text=False
    )

    story = result[0]["generated_text"].strip()

    forbidden_words = [
        "murder", "kill", "blood", "gun", "knife", "weapon",
        "gangster", "crime", "death", "dead", "horror",
        "scary", "violence", "violent"
    ]

    story_lower = story.lower()

    for word in forbidden_words:
        if re.search(r"\b" + word + r"\b", story_lower):
            return None

    return story


# Function 3: Story to Audio
def story2audio(story):
    tts = gTTS(text=story, lang="en")

    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)

    return audio_file.name


# Function 4: Main App
def main():
    st.set_page_config(
        page_title="Storytelling App",
        page_icon="📖"
    )

    st.title("Storytelling App")
    st.write("Upload a picture and create a short audio story for children.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        if st.button("Generate Story"):
            with st.spinner("Reading the image..."):
                scenario = img2text(image_path)

            st.subheader("Image Description")
            st.write(scenario)

            with st.spinner("Writing the story..."):
                story = text2story(scenario)

            if story is None:
                st.warning("The story may not be suitable
