

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
    caption = result[0]["generated_text"].strip()

    return caption


# Function 2: Text to Story
def text2story(scenario):
    if "story_model" not in st.session_state:
        st.session_state.story_model = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a children's story writer. "
                "Write safe, warm, simple stories for children aged 3 to 10. "
                "Return only the story."
            )
        },
        {
            "role": "user",
            "content": (
                f"Image description: {scenario}\n"
                "Write one complete short story based only on this image description. "
                "The story should be 50 to 100 words. "
                "Do not add unrelated people, places, or events. "
                "Do not write a title. Do not write a summary."
            )
        }
    ]

    result = st.session_state.story_model(
        messages,
        max_new_tokens=130,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15
    )

    generated_text = result[0]["generated_text"]

    # For chat-style output
    if isinstance(generated_text, list):
        story = generated_text[-1]["content"]
    else:
        story = generated_text

    # Clean output
    story = story.strip()
    story = story.replace("Story:", "").strip()
    story = story.replace("Title:", "").strip()
    story = story.replace("Summary:", "").strip()
    story = re.sub(r"\s+", " ", story)

    # Remove possible prompt repetition
    bad_phrases = [
        "image description:",
        "write one complete",
        "do not write",
        "the story should",
        "return only"
    ]

    for phrase in bad_phrases:
        if phrase in story.lower():
            story = story.split(phrase)[0].strip()

    # Keep story within 100 words
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
        last_period = story.rfind(".")
        if last_period != -1:
            story = story[:last_period + 1]
        else:
            story += "."

    # If the model gives an incomplete sentence, add a simple ending
    if story and not story.endswith((".", "!", "?")):
        story += "."

    # Safety check
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

            with st.spinner("Reading the image..."):
                scenario = img2text(image_path)

            st.subheader("Image Description")
            st.write(scenario)

            with st.spinner("Generating story..."):
                story = text2story(scenario)

            st.subheader("Generated Story")
            st.write(story)

            with st.spinner("Generating audio..."):
                audio_path = story2audio(story)

            st.subheader("Audio Story")
            st.audio(audio_path)


if __name__ == "__main__":
    main()
