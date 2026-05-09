


   
  
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

    if isinstance(generated_text, list):
        story = generated_text[-1]["content"]
    else:
        story = generated_text

    story = story.strip()
    story = story.replace("Story:", "").strip()
    story = story.replace("Title:", "").strip()
    story = story.replace("Summary:", "").strip()
    story = re.sub(r"\s+", " ", story)

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
        last_period = story.rfind(".")
        if last_period != -1:
            story = story[:last_period + 1]
        else:
            story += "."

    if story and not story.endswith((".", "!", "?")):
        story += "."

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
        page_title="Magic Storytelling App",
        page_icon="🌈",
        layout="centered"
    )

    # CSS style
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFF7D6 0%, #FFE4F3 45%, #DDF3FF 100%);
        }

        .main-title {
            text-align: center;
            font-size: 46px;
            font-weight: 800;
            color: #FF7A59;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #5D5D5D;
            margin-bottom: 30px;
        }

        .step-card {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 25px;
            box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.08);
            text-align: center;
            margin-bottom: 20px;
        }

        .story-card {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 25px;
            box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.10);
            font-size: 19px;
            line-height: 1.8;
            color: #444444;
            margin-top: 15px;
        }

        .description-card {
            background-color: #FFF2CC;
            padding: 18px;
            border-radius: 20px;
            font-size: 17px;
            color: #555555;
            margin-top: 10px;
        }

        .stButton > button {
            background-color: #FF9F45;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 14px 28px;
            font-size: 20px;
            font-weight: bold;
            width: 100%;
        }

        .stButton > button:hover {
            background-color: #FF7A59;
            color: white;
        }

        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 18px;
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title
    st.markdown(
        "<div class='main-title'>🌈 Magic Storytelling App 📖</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Upload a picture and turn it into a gentle audio story for children.</div>",
        unsafe_allow_html=True
    )

    # Three simple steps
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<div class='step-card'>📷<br><b>Step 1</b><br>Upload a picture</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            "<div class='step-card'>✨<br><b>Step 2</b><br>Create a story</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            "<div class='step-card'>🔊<br><b>Step 3</b><br>Listen to audio</div>",
            unsafe_allow_html=True
        )

    uploaded_file = st.file_uploader(
        "Choose an image for your story",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.markdown("### 🖼️ Your Picture")

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        st.write("")

        if st.button("✨ Create My Story and Audio"):

            with st.spinner("Reading the picture..."):
                scenario = img2text(image_path)

            st.markdown("### 🔍 Image Description")
            st.markdown(
                f"<div class='description-card'>{scenario}</div>",
                unsafe_allow_html=True
            )

            with st.spinner("Writing a magical story..."):
                story = text2story(scenario)

            st.markdown("### 📖 Generated Story")
            st.markdown(
                f"<div class='story-card'>{story}</div>",
                unsafe_allow_html=True
            )

            with st.spinner("Turning the story into audio..."):
                audio_path = story2audio(story)

            st.markdown("### 🔊 Audio Story")
            st.audio(audio_path)

            st.success("Your story is ready! Enjoy listening. 🌟")
            st.balloons()


if __name__ == "__main__":
    main()
