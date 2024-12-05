import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
## Handles data, recommendation generation, feedback interpretation, and user profiles

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
def load_data():
    """
    Load and return all relevant data

    Returns:
        dictionary containing all data
    """
    with open("courses.json", "r") as file:
        courses = json.load(file)
        current_courses = courses["current"]
        past_courses = courses["past"]
    loaded_embeddings = np.load('embeddings.npz', allow_pickle=True)
    current_embeddings = loaded_embeddings['current_courses']
    past_embeddings = loaded_embeddings['prev_courses']
    title_embeddings = dict(loaded_embeddings['titles'].item())
    intent_embeddings = dict(loaded_embeddings['intent'].item())

    # Create a dictionary containing all data
    data = {
        'current_courses': current_courses, 
        'past_courses': past_courses,
        'current_embeddings': current_embeddings,
        'past_embeddings': past_embeddings,
        'title_embeddings': title_embeddings,
        'intent_embeddings': intent_embeddings
        }
    return data


### JUST FOR TESTING:
def select_random():
    data_list = load_data()['current_courses']
    response = random.choice(data_list)['title']
    return response