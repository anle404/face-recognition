import os
import numpy as np
import joblib
from collections import defaultdict
from PIL import Image
from face_crop_plus import Cropper
from model import calculate_distance

def add_person(name, imgs, encoder):
    folder_path = f"./static/raw_images/{name}"
    cropped_path = f"./static/known_faces/{name}"

    if os.path.exists(folder_path):
        return "Username already registered"
    
    if os.path.exists('./embeddings_dict.pkl'):
        embeddings_dict = load_emb_dict()
    else:
        embeddings_dict = defaultdict(list)
    
    store_images(name, imgs, folder_path, cropped_path)

    for filename in os.listdir(cropped_path):
        cropped_img_path = os.path.join(cropped_path, filename)
        cropped_img = Image.open(cropped_img_path)
        cropped_img = np.asarray(cropped_img)

        emb = encode_img(cropped_img, encoder)
        embeddings_dict[name].append(emb)
        write_emb_dict(embeddings_dict)

    return ""

def write_emb_dict(emb_dict):
    file_path = "./embeddings_dict.pkl"
    with open(file_path, "wb") as f:
        joblib.dump(emb_dict, f, compress=0)
    return

def load_emb_dict():
    file_path = "./embeddings_dict.pkl"
    with open(file_path, "rb") as f:
        emb_dict = joblib.load(f)
    return emb_dict

def store_images(name, imgs, folder_path, cropped_path):
    img_id = 1
    for img in imgs: 
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{name}_{img_id}.jpg"
        img_path = os.path.join(folder_path, filename)
        saved_img = Image.fromarray(img)
        saved_img.save(img_path)

        face_crop(folder_path, cropped_path)
        img_id += 1
    return


def face_crop(input_dir, output_dir):
    cropper = Cropper(
        output_size=(64, 64),
        output_format="jpg",
        face_factor=0.8
    )

    cropper.process_dir(input_dir, output_dir)

def encode_img(img, encoder):
    img = np.expand_dims(img, axis=0)
    emb = encoder(img)

    return emb.numpy()

def get_person_name(img, encoder):

    if os.path.exists('./embeddings_dict.pkl'):
        embeddings_dict = load_emb_dict()
    else:
        return "No Registration Entry"
    
    folder_path = f'./static/checkin_raw_images'
    cropped_path = './static/checkin'

    img_id = {len(os.listdir(folder_path)) + 1}

    img_dir = f'{folder_path}/{img_id}'
    os.makedirs(img_dir, exist_ok=True)

    filename = f"{img_id}.jpg"

    img_path = os.path.join(img_dir, filename)
    saved_img = Image.fromarray(img)
    saved_img.save(img_path)

    face_crop(img_dir, cropped_path)

    cropped_img_path = os.path.join(cropped_path, filename)
    cropped_img = Image.open(cropped_img_path)
    cropped_img = np.asarray(cropped_img)

    emb_new = encode_img(cropped_img, encoder)

    recognize_name = ""
    lowest_dist = 1000
    for key in embeddings_dict.keys():
        for emb in embeddings_dict[key]:
            distance = calculate_distance(emb, emb_new)
            print(distance)
            if distance < lowest_dist and distance < 0.5:
                recognize_name = key
                lowest_dist = distance
            else:
                continue
    
    return recognize_name



    
    

 
    