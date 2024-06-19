import streamlit as st
import requests
import tempfile
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai

load_dotenv(find_dotenv())
# HUGGINGFACE_HUB_API=os.getenv("HUGGINGFACE_HUB_API")
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# dotenv.load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Configure the GenerativeAI API key
genai.configure(api_key="AIzaSyCLt-nnDsWdW7tJQLTD5Wiq54tW6_Wm55w")

# Create a GenerativeModel instance
model = genai.GenerativeModel('gemini-pro')

# image to text
def imgToText(file_path):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(file_path)[0]['generated_text']
    return text


def generate_text_with_gemini(prompt):


  try:
    # Create a GenerativeModel instance
    model = genai.GenerativeModel('gemini-pro')
    # Generate content using the model
    response = model.generate_content(prompt)

    if response and hasattr(response, 'text'):
      return response.text
    else:
      return response.text  # Handle cases where no text is generated

  except Exception as e:
    print(f"An error occurred: {str(e)}")
    return None  # Handle errors
# LLM
def generate_story(scenario):
    template = """
            You are a storyteller. Tell me a short story (no more than 100 words) based on the following scenario:

            CONTEXT: {scenario}

             STORY:
            """

    prompt = template.format(scenario=scenario)

    #response = model.generate_content(user_prompt)
    # prompt = PromptTemplate(template=template, input_variables=["scenario"])
    #story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo"), prompt=prompt, verbose=True)
    #story = story_llm.predict(scenario=scenario)
    generated_text = generate_text_with_gemini(prompt)
    return generated_text

def textToSpeech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer " + HUGGINGFACEHUB_API_TOKEN}
    payload = {"inputs": story}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open("story.flac", "wb") as f:
        f.write(response.content)

def generate_story_and_play_audio(image):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(image.read())
        temp_file_path = temp_file.name

    scenario = imgToText(temp_file_path)
    st.write("Scenario")
    st.write(scenario)
    story = generate_story(scenario)
    st.write("Story")
    st.write(story)
    textToSpeech(story)

    os.unlink(temp_file_path)  # Remove the temporary file
    return "story.flac"

st.title("Generate Story from Image")
st.sidebar.markdown("### Upload an image:")
uploaded_image = st.sidebar.file_uploader(label="Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.audio(generate_story_and_play_audio(uploaded_image), format="audio/flac")
    st.write()
    st.write()




