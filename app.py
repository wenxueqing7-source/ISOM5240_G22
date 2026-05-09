



  
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
            model="HuggingFaceTB/SmolLM2-360M-Instruct"
        )

    prompt = f"""
You are a children's storyteller.

Image description:
{scenario}

Write one short story for children aged 3 to 11.

Rules:
- The story must be based on the image description.
- Do not add unrelated topics.
- Do not add marriage, war, crime, death, weapons, or scary events.
- Do not write a summary.
- Do not mention "Summary".
- Use simple English.
- Make the story warm, gentle, and positive.
- The story should feel like a real bedtime story.
- Around 100 words.

Story:
"""

    forbidden_words = [
        "murder", "kill", "blood", "gun", "knife", "weapon",
        "gangster", "crime", "death", "dead", "horror",
        "scary", "violence", "violent", "war", "marriage"
    ]

    for attempt in range(3):
        result = st.session_state.story_model(
            prompt,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.55,
            top_p=0.85,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            return_full_text=False
        )

        story = result[0]["generated_text"].strip()

        # Remove unwanted extra sections
        story = story.split("Summary:")[0].strip()
        story = story.split("Image description:")[0].strip()
        story = story.split("Rules:")[0].strip()

        story_lower = story.lower()

        unsafe = False
        for word in forbidden_words:
            if re.search(r"\b" + word + r"\b", story_lower):
                unsafe = True
                break

        if not unsafe and len(story.split()) >= 50:
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

            with st.spinner("Reading the image..."):
                scenario = img2text(image_path)

            st.subheader("Image Description")
            st.write(scenario)

            with st.spinner("Generating story..."):
                story = text2story(scenario)

            if story is None:
                st.warning(
                    "The app could not generate a suitable story this time. "
                    "Please try again or upload another image."
                )
                st.stop()

            st.subheader("Generated Story")
            st.write(story)

            with st.spinner("Generating audio..."):
                audio_path = story2audio(story)

            st.subheader("Audio Story")
            st.audio(audio_path)


if __name__ == "__main__":
    main()
