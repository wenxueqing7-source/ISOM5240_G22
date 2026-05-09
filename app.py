

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
    scenario = result[0]["generated_text"]
    return scenario


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
- Suitable for children aged 3 to 11.
- No violence.
- No death.
- No weapons.
- No scary content.
- Use simple words.
- Make the story warm, happy, and positive.
- The story should be about kindness, friendship, curiosity, or imagination.

Story:
"""

    forbidden_words = [
        "murder", "kill", "blood", "gun", "knife", "weapon",
        "gangster", "crime", "death", "dead", "horror",
        "scary", "violence", "violent"
    ]

    # Try to generate a safe story up to 3 times
    for i in range(3):
        result = st.session_state.story_model(
            prompt,
            max_new_tokens=140,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            return_full_text=False
        )

        story = result[0]["generated_text"].strip()

        story_lower = story.lower()
        unsafe = False

        for word in forbidden_words:
            if re.search(r"\b" + word + r"\b", story_lower):
                unsafe = True
                break

        if not unsafe:
            return story

    return None


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
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        if st.button("Generate Story and Audio"):

            # Stage 1: Image to Text
            with st.spinner("Reading the image..."):
                scenario = img2text(image_path)

            st.subheader("Image Description")
            st.write(scenario)

            # Stage 2: Text to Story
            with st.spinner("Generating story..."):
                story = text2story(scenario)

            if story is None:
                st.warning(
                    "The app could not generate a safe story this time. "
                    "Please try again or upload another image."
                )
                st.stop()

            st.subheader("Generated Story")
            st.write(story)

            # Stage 3: Story to Audio
            with st.spinner("Generating audio..."):
                audio_path = story2audio(story)

            st.subheader("Audio Story")
            st.audio(audio_path)


if __name__ == "__main__":
    main()
