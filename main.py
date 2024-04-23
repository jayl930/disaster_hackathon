from dotenv import load_dotenv
import os
import json
import requests

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from pymongo.mongo_client import MongoClient
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account
import urllib.parse

import subprocess
from PIL import Image
import gradio as gr
from transformers import pipeline
import numpy as np

from shapely.geometry import MultiPolygon, Polygon
from shapely import wkt
from PIL import Image, ImageDraw

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
api_key = os.getenv("GOOGLE_MAPS_API_KEY")
uri = os.getenv("MONGO")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    openai_api_version="2024-03-01-preview",
    temperature=0.1,
)


asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


class EmergencyInfo(BaseModel):
    """Information Extracted from Emergency Situation"""

    name: Optional[str] = Field(
        None, description="Name of the person involved in the emergency"
    )
    appearance: Optional[str] = Field(
        None, description="Description of the person's appearance"
    )
    address: Optional[str] = Field(
        None, description="Address where the emergency is occurring"
    )
    situation: Optional[str] = Field(
        None,
        description="Detailed description of the emergency situation, described in third person",
    )


def geocode_address(api_key, address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return response.status_code, response.text

    info = response.json()
    if info["status"] == "OK" and info["results"]:
        location = info["results"][0]["geometry"]["location"]
        # coordinates = {"lat": location["lat"], "lng": location["lng"]}
        return location
    else:
        return "No location found or error in response."


def save_emergency_details(emergency_details):
    # Convert dictionary to JSON string and save to file
    with open("emergency_details.json", "w") as file:
        json.dump(emergency_details, file, indent=4)
    print("Emergency details saved to JSON.")


def handle_emergency_situation(question):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Listen carefully to the user's explanation of their emergency situation. Extract the following details and format them into simple string responses:
                - 'name': The name of the person.
                - 'appearance': A description of the person's appearance.
                - 'address': The address where assistance is needed.
                - 'situation': A detailed description of the situation facing the person, written in third person.
                Format the information neatly as a JSON structure with each detail clearly noted as individual strings. If any detail is missing, use 'None' for that field. Do not make assumptions to fill in missing information.
                """,
            ),
            ("human", "{question}"),
        ]
    )
    parser = JsonOutputParser(pydantic_object=EmergencyInfo)
    chain = prompt | llm | parser
    response = chain.invoke(question)
    address = response["address"]
    geocode_result = geocode_address(api_key, address)
    emergency_details = {
        "name": response["name"],
        "appearance": response["appearance"],
        "address": response["address"],
        "situation": response["situation"],
        "latitude": geocode_result["lat"],
        "longitude": geocode_result["lng"],
    }
    print(emergency_details)
    save_emergency_details(emergency_details)
    # if isinstance(geocode_result, tuple):
    #     print(f"Error {geocode_result[0]}: {geocode_result[1]}")
    # elif isinstance(geocode_result, str):
    #     print(geocode_result)
    # else:
    return geocode_result


def retrieve_image(address):
    coords = handle_emergency_situation(address)
    latitude = coords["lat"]
    longitude = coords["lng"]
    client = MongoClient(uri)
    db = client["ncsa"]
    images_collection = db["ncsa"]

    point = {
        "type": "Point",
        "coordinates": [longitude, latitude],  # Ensure longitude is first
    }
    image = images_collection.find_one(
        {"bounding_box": {"$geoIntersects": {"$geometry": point}}}
    )
    if image:
        print("Image found:", image["name"])
        bounding_box_coords = image["bounding_box"]["coordinates"][0]
        northwest = bounding_box_coords[0]  # Min longitude, Max latitude
        northeast = bounding_box_coords[1]  # Max longitude, Max latitude
        southeast = bounding_box_coords[2]  # Max longitude, Min latitude
        southwest = bounding_box_coords[3]  # Min longitude, Min latitude

        return {
            "pre_url": image["pre"],
            "post_url": image["post"],
            "northwest": northwest,
            "northeast": northeast,
            "southeast": southeast,
            "southwest": southwest,
        }
    else:
        print("No image found containing the point.")
        return None


def get_credentials():
    json_file_path = "ncsa-hackathon-f88e88a84517.json"
    with open(json_file_path, "r") as file:
        credentials_dict = json.load(file)
    return service_account.Credentials.from_service_account_info(credentials_dict)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def get_images(address, download_path):
    # Call retrieve_image to get the image URLs and other details
    image_data = retrieve_image(address)
    if image_data is None:
        print("No image found for the specified address.")
        return

    bucket_name = "ncsa-hackthon"
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    def get_blob_name_from_url(url):
        parsed_url = urllib.parse.urlparse(url)
        path_parts = parsed_url.path.split("/")
        blob_name = "/".join(path_parts[2:])  # Adjust if bucket structure is different
        return blob_name

    # Retrieve the blob names from URLs
    pre_blob_name = get_blob_name_from_url(image_data["pre_url"])
    post_blob_name = get_blob_name_from_url(image_data["post_url"])

    # Get filenames from the blob names
    pre_file_name = pre_blob_name.split("/")[-1]
    post_file_name = post_blob_name.split("/")[-1]

    # Construct local file paths for download
    pre_path = os.path.join(download_path, pre_file_name)
    post_path = os.path.join(download_path, post_file_name)

    # Download the images to the specified path
    download_blob(bucket_name, pre_blob_name, pre_path)
    download_blob(bucket_name, post_blob_name, post_path)

    return (
        image_data["northwest"],
        image_data["northeast"],
        image_data["southeast"],
        image_data["southwest"],
    )


def latlng_to_pixels(lat, lng, img_bounds, img_size):
    min_lat, max_lat, min_lng, max_lng = img_bounds

    lat_scale = img_size[1] / (max_lat - min_lat)
    lng_scale = img_size[0] / (max_lng - min_lng)

    x = (lng - min_lng) * lng_scale
    y = (max_lat - lat) * lat_scale
    return (x, y)


def classify_damage(rgb):
    red = (255, 0, 0)
    blue = (0, 0, 255)
    distance_to_red = (
        (rgb[0] - red[0]) ** 2 + (rgb[1] - red[1]) ** 2 + (rgb[2] - red[2]) ** 2
    ) ** 0.5
    distance_to_blue = (
        (rgb[0] - blue[0]) ** 2 + (rgb[1] - blue[1]) ** 2 + (rgb[2] - blue[2]) ** 2
    ) ** 0.5

    if distance_to_red < distance_to_blue:
        if rgb[1] < 60:
            return "destroyed"
        elif rgb[1] < 120:
            return "major-damage"
        else:
            return "minor-damage"
    else:
        return "no-damage"


def get_bounds_and_draw(img_bounds, lat, lng, img_path):
    img = Image.open(img_path)
    img_size = img.size
    pixel_coords = latlng_to_pixels(lat, lng, img_bounds, img_size)
    path = img_path.split("/")[-1]
    draw = ImageDraw.Draw(img)
    radius = 20
    color = "red"
    draw.ellipse(
        (
            pixel_coords[0] - radius,
            pixel_coords[1] - radius,
            pixel_coords[0] + radius,
            pixel_coords[1] + radius,
        ),
        outline=color,
        width=3,
    )
    color_at_pixel = img.getpixel((int(pixel_coords[0]), int(pixel_coords[1])))
    damage_category = classify_damage(color_at_pixel)
    modified_img_path = "marked/image.png"
    img.save(modified_img_path)
    return damage_category


def process_text_and_images(text):
    nw, ne, se, sw = get_images(text, "data")
    subprocess.run(["python", "predictclimax_cls.py", "0"])
    image_dir = "pred34_cls_"
    with open("emergency_details.json", "r") as file:
        emergency_details = json.load(file)

    lat = emergency_details.get("latitude")
    lng = emergency_details.get("longitude")

    if lat is None or lng is None:
        print("Latitude or longitude not found in JSON.")
        return None
    img_bounds = (sw[1], ne[1], nw[0], se[0])
    img_path = os.path.join(image_dir, os.listdir(image_dir)[1])
    classify = get_bounds_and_draw(img_bounds, lat, lng, img_path)
    emergency_details["classify"] = classify
    txt_path = "pred34_cls_/destruction_percentages.txt"
    with open(txt_path, "r") as file:
        destroyed_content = file.readline().strip()
    emergency_details["destroyed"] = destroyed_content

    with open("emergency_details.json", "w") as file:
        json.dump(emergency_details, file, indent=4)
    image = Image.open("marked/image.png")
    return image


def transcribe(audio):
    sampling_rate, audio_numpy = audio
    audio_numpy = audio_numpy.astype(np.float32)
    if np.max(np.abs(audio_numpy)) > 0:
        audio_numpy /= np.max(np.abs(audio_numpy))
    transcript = asr_pipeline({"raw": audio_numpy, "sampling_rate": sampling_rate})[
        "text"
    ]
    return transcript


def full_workflow(audio):
    transcript = transcribe(audio)
    process_text_and_images(transcript)
    image = Image.open("marked/image.png")
    return transcript, image


with gr.Blocks() as demo:
    with gr.Row():
        audio_input = gr.Audio(type="numpy", label="Record your speech")
        transcribe_button = gr.Button("Transcribe Audio")
    transcription_output = gr.Textbox(
        label="Transcription", placeholder="Transcribed text will appear here..."
    )
    process_button = gr.Button("Process Transcription")
    image_output = gr.Image(label="Destroyed Buildings")

    transcribe_button.click(
        transcribe, inputs=audio_input, outputs=transcription_output
    )
    process_button.click(
        process_text_and_images,
        inputs=transcription_output,
        outputs=[image_output],
    )

demo.launch()
