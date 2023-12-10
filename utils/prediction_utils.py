import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from utils.matplotlib_utils import plot_image
from utils.models_utils import load_model
from models.resNet import ResNet9


# Function that gets image from link #
def get_image_from_url(url: str) -> Image:
    '''
    Parameters
    ----------
    url : str

    Returns
    ----------
    Pillow Image.

    Notes
    ----------
    Gathers an image with get request and converts it to Pillow Image.
    '''
    try:
        # Make a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Read the image data from the response content
            image_data = response.content

            # Create a BytesIO object to handle the image data
            image_buffer = BytesIO(image_data)

            # Open the image using PIL (Pillow)
            image = Image.open(image_buffer)

            # You can now work with the 'image' object (e.g., display or save it)
            return image
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Function that predicts image using  #
def predict_img(img_url: str, device:torch.device) -> str:
    '''
    Parameters
    ----------
    url : str
    device : torch.device

    Returns
    ----------
    String.

    Notes
    ----------
    Takes Url for an image and predicts the image using the ResNet9 model.
    '''

    # Get image from url and resize it to model data size #
    image = get_image_from_url(url=img_url)
    image = image.resize((32, 32))
    image_arr = np.transpose(np.array(image),[2,0,1])

    # Normalize and plot image #
    image_arr = image_arr / 255.0
    plot_image(image_arr)

    # Convert image to tensor and predict #
    image_arr = torch.unsqueeze(((torch.from_numpy(image_arr).to(torch.float32)).to(device)),0)
    recon_img_model:ResNet9 = load_model(ResNet9(3,100),"models/trained_models/resNet.pt")
    recon_img_model.eval()
    prediction = recon_img_model(image_arr)

    map = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
    ]

    # Check label map and return label index based on prediction #
    index = np.argmax(prediction.detach().numpy())
    label = map[index]

    return label