
 
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
            "text2text-generation",
            model="google/flan-t5-small"
        )

    prompt = f"""
    Write a short bedtime story for children aged 3 to 11.

    Image description: {scenario}

    Rules:
    No violence, no murder, no weapons, no gangsters, no death, no scary content.
    Use simple, warm, positive language.
    
    Around 100 words.
    """

    result = st.session_state.story_model(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    story = result[0]["generated_text"]
    return story


# Function 3: Story to Audio
def story2audio(story):
    tts = gTTS(text=story, lang="en")
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)
    return audio_file.name


# Simple safety check
def is_safe(text):
    forbidden_words = [
        "murder", "kill", "blood", "gun", "knife", "weapon",
        "gangster", "crime", "death", "dead", "horror",
        "scary", "violence", "violent"
    ]

    text = text.lower()

    for word in forbidden_words:
        if re.search(r"\b" + word + r"\b", text):
            return False

    return True


# Function 4: Main App
def main():
    st.set_page_config(
        page_title="Storytelling App",
        page_icon="🦜"
    )

    st.header("Turn Your Image into an Audio Story")

    uploaded_file = st.file_uploader(
        "Select an Image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        if st.button("Generate Story and Audio"):

            st.text("Processing image...")
            scenario = img2text(image_path)
            st.write("**Scenario:**")
            st.write(scenario)

            st.text("Generating story...")
            story = text2story(scenario)

            if not is_safe(story):
                st.warning("The story may not be safe for children. Please try again.")
                st.stop()

            st.write("**Story:**")
            st.write(story)

            st.text("Generating audio...")
            audio_path = story2audio(story)

            st.audio(audio_path)


if __name__ == "__main__":
    main()
