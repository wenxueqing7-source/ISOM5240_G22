



  

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

    result = st.session_state.img_model(
        image_path,
        max_new_tokens=50
    )

    caption = result[0]["generated_text"].strip()
    return caption


# Function 2: Text to Story
def text2story(scenario):
    if "story_model" not in st.session_state:
        st.session_state.story_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    prompt = f"""
Write a short children's story based ONLY on this image description:

Image description: {scenario}

Rules:
- The story must be 50 to 100 words.
- The story must stay consistent with the image description.
- Do not add unrelated topics, people, places, or events.
- Use simple English for children aged 3 to 10.
- Make the story warm, positive, gentle, and imaginative.
- Return only the story, no title, no summary.

Story:
"""

    result = st.session_state.story_model(
        prompt,
        max_new_tokens=120,
        num_beams=4,
        do_sample=False
    )

    story = result[0]["generated_text"].strip()

    # Clean output
    story = story.replace("Story:", "").strip()
    story = story.replace("Summary:", "").strip()
    story = re.sub(r"\s+", " ", story)

    # Make sure the story is connected to the image
    if scenario.lower() not in story.lower():
        story = f"In the picture, {scenario}. " + story

    # Limit to around 100 words
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
        if not story.endswith("."):
            story += "."

    # Simple safety check
    forbidden_words = [
        "murder", "kill", "blood", "gun", "knife", "weapon",
        "gangster", "crime", "death", "dead", "horror",
        "violence", "violent", "war"
    ]

    story_lower = story.lower()

    for word in forbidden_words:
        if re.search(r"\b" + word + r"\b", story_lower):
            return "The app generated an unsuitable story. Please click the button again or try another image."

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

            st.subheader("Generated Story")
            st.write(story)

            # Stage 3: Story to Audio
            with st.spinner("Generating audio..."):
                audio_path = story2audio(story)

            st.subheader("Audio Story")
            st.audio(audio_path)


if __name__ == "__main__":
    main()
