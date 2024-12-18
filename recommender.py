import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from chatbot_variables import intent_replies, module_dict
import string

## Handles data, recommendation generation, feedback interpretation, and user profiles
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#def load_data():
#    """
#    Load and return all relevant data#
#
 #   Returns:
 #       dictionary containing all data
 #   """
#    with open("courses.json", "r") as file:
#        courses = json.load(file)
#        current_courses = courses["current"]
#        past_courses = courses["past"]
#    loaded_embeddings = np.load('embeddings.npz', allow_pickle=True)
#    current_embeddings = loaded_embeddings['current_courses']
#    past_embeddings = loaded_embeddings['prev_courses']
#    title_embeddings = dict(loaded_embeddings['titles'].item())
#    intent_embeddings = dict(loaded_embeddings['intent'].item())

    # Create a dictionary containing all data
#    data = {
#        'current_courses': current_courses, 
#        'past_courses': past_courses,
#        'current_embeddings': current_embeddings,
#        'past_embeddings': past_embeddings,
#        'title_embeddings': title_embeddings,
#        'intent_embeddings': intent_embeddings
#        }
#    return data


# Load the course and embedding data (created and saved in data_prep.py)
with open("courses.json", "r") as file:
    courses = json.load(file)
    current_courses = courses["current"]
    past_courses = courses["past"]
    current_attributes = courses["current_attr"]
    past_attributes = courses["past_attr"]
loaded_embeddings = np.load('embeddings.npz', allow_pickle=True)
current_embeddings = loaded_embeddings['current_courses']
past_embeddings = loaded_embeddings['prev_courses']
title_embeddings = dict(loaded_embeddings['titles'].item())
intent_embeddings = dict(loaded_embeddings['intent'].item())


def get_data():### PROB USELESS
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

def get_five_courses(): ### JUST FOR TESTING
    return current_courses[7:12]

def get_past_title(idx): ### JUST FOR TESTING
    """
    Returns the title of the course with the given index in the list of past courses

    Parameter:
        idx (int): index of the course
    Returns:
        title (str) of the course
    """
    if isinstance(idx, int) and idx < len(past_courses):
        #print(f"*** Found past course {past_courses[idx]['title']} at index {idx}")
        return past_courses[idx]['title']
    else:
        print(f"*** get_past_title(): NO PAST COURSE FOUND FOR IDX {idx}")
        return None

def get_current_title(idx):### JUST FOR TESTING
    """
    Returns the title of the course with the given index in the list of current courses

    Parameter:
        idx (int): index of the course
    Returns:
        title (str) of the course
    """
    if isinstance(idx, int) and idx < len(current_courses):
        #print(f"*** Found current course {current_courses[idx]['title']} at index {idx}")
        return current_courses[idx]['title']
    else:
        print(f"*** get_current_title(): NO CURRENT COURSE FOUND FOR IDX {idx}")
        return None

def get_past_idx(title):
    """
    Returns the index of a past course given it's title
    
    Parameter:
        title (str): title of the course
    Returns:
        index (int)
    """

    idx = [c['title'] for c in past_courses].index(title)
    #print(f"***get_past_idx(): title: {title} -- idx: {idx}")
    #print(f"*** in get_past_idx: Found index '{idx}' for '{title}' -- title by idx: {past_courses[idx]['title']}")
    if isinstance(idx, int):
        #print(f"*** Found past idx {idx} for title {title}")
        return idx
    else:
        print(f"*** get_past_idx(): NO PAST IDX FOUND FOR TITLE {title}")
        return None

def get_details(idx):
    """
    Returns dictionary with all attributes of a current course

    Parameter:
        idx: index of the course to return
    Returns:
        all attributes of the course
    """
    return current_courses[idx]

###--- Handle Intent Detection ---###

def detect_intent(user_input, last_recommendations):
    """
    Detects the intent of a user's input

    Parameters:
        user_input (str): the user's input
        last_recommendations (list): list of the last recommended courses
    Returns:
        detected_intent: the intent it detected
        intent_replies[detected_intent] or chatbot_reply: the chatbots reply based on the intent it detected
        detected_courses: tuple (course index, x) with x being either the similarity (float) between the title and user input (if detected_intent == reference) or the feedback for the course ('liked' or 'disliked') (if detected_intent == feedback)
    """
    print("***detect_intent(): Detecting intent...")
    intent_similarities = {'free_description': 0.0, 'liked_course_reference': 0.0, 'feedback': 0.0}
    user_embedding = model.encode([user_input])[0]
    detected_intent = "other"  # Default intent

    # First check if the user has explicitly stated the intent at the beginning of the message
    if user_input.lower().startswith("free:"):
        return "free_description", intent_replies["free_description"], []
    elif user_input.lower().startswith("ref:"):
        chatbot_reply, intent_result = check_intent("liked_course_reference", user_input, user_embedding, last_recommendations)
        if chatbot_reply != "":
            return "liked_course_reference", chatbot_reply, intent_result
        else:
            return "other", "I'm sorry, but I don't understand which course you are referring to. Please make sure that the name is correct."
    elif user_input.lower().startswith("feedback:"):
        chatbot_reply, intent_result = check_intent("feedback", user_input, user_embedding, last_recommendations)
        return "feedback", chatbot_reply, intent_result
    
    # If no intent was given, select the one which is most similar to the input
    else:
        # Compare user input with each intent category
        for intent, examples in intent_embeddings.items():
            # Only allow feedback if there have been recommendations
            if intent == 'feedback' and len(last_recommendations) == 0:
                intent_similarities[intent] = 0.0
                continue
            similarity = float(cosine_similarity([user_embedding], examples).max())
            intent_similarities[intent] = similarity

        # Get the intent with the highest similarity
        sorted_intents = dict(sorted(intent_similarities.items(), key=lambda item: item[1])[::-1])
        #print(f"***Before sorting: {intent_similarities}\n***After sorting: {sorted_intents}")
        print(f"***detect_intent(): Intent similarities: {sorted_intents}")
        it = iter(sorted_intents.items())

        for detected_intent in it:
            #detected_intent = next(it)
            print(f"***detect_intent(): Checking intent '{detected_intent}'!")
        
            # If the similarity score of the intent with the highest similarity to the input is below the threshold, return "other"
            if detected_intent[1] < 0.5 and detected_intent[1] != "nonsense":
                return "other", intent_replies["other"], []
            # If it is one of the intents that just return the corresponding reply, there is no need to ckeck the, directly return
            elif detected_intent[0] in ["greeting", "free_description", "nonsense", "other"]:
                return detected_intent[0], intent_replies[detected_intent[0]], []
            chatbot_reply, intent_result = check_intent(detected_intent, user_input, user_embedding, last_recommendations)

            # If the chatbot reply is not empty, return the current detected intent; otherwise, the loop continues to the next intent
            if chatbot_reply != "":
                return detected_intent[0], chatbot_reply, intent_result
            
    return "other", intent_replies["other"], []


def check_intent(detected_intent, user_input, user_embedding, last_recommendations):
    if detected_intent[0] == "liked_course_reference":
        # First check if a title is spelled out exactly (only one reference per message allowed)
        best_fit = ("", 0.0)
        all_titles = [c['title'] for c in past_courses]
        for title in all_titles:
            ##### MAYBE EHER VARIANTE NUTZEN, DIE AUCH LEICHTE ABWANDLUNGEN (z.B. DURCH TIPPFEHLER) ERKENNT
            ####### DANN VLLT LISTE (ODER DICT MIT SIMILARITY-WERTEN) ERSTELLEN & DA ALLE TITEL, DIE ÄHNLICH GENUG SIND, DRIN SPEICHERN & AM ENDE BEST FIT RETURNEN
            ### IST DAS ÜBERHAUPT NOTWENDIG???
            if title in user_input:
                idx = get_past_idx(title)
                print(f"***detect_intent(): Found EXACT Title: {title} (by index {idx}: {past_courses[idx]['title']})")
                #detected_courses.append((title, 1.0))  ### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
                return intent_replies[detected_intent[0]], [(idx, 1.0)]
            
        
        # Substract the difference between the similarity of liked course references and free description from the threshold  ### OR BETTER JUST SIM OF REFERENCE??
        # -> the more certain it is to be a reference, the lower is the threshold for the title similarity
        ### NICHT SICHER, OB DAS WIRKLICH GUT IST (WEGEN NONSENSE)
        #title_threshold = 0.55 - (intent_similarities['liked_course_reference'] - intent_similarities['free_description'])
        #print(f"***detect_intent(): Title threshold = {title_threshold}")

        # Set the threshold lower if the similarity for the intent was set to 1.0
        if detected_intent[1] >= 1.0:
            title_threshold = 0.2
        else:
            title_threshold = 0.5

        # Check if the input is similar enough to any of the titles
        for title, title_emb in title_embeddings.items():  ### KANN ICH SONST AUCH ÜBER INDEX MACHEN -> BRAUCHE DANN TITLE NICHT MEHR IN TITLE_EMBEDDINGS -> IST DANN NUR NOCH LIST STATT DICT
            title_sim = cosine_similarity([user_embedding], [title_emb])
            #print(f"{title}: {title_sim}")

            ## JUST TESTING 
            if title_sim > 0.3:
                print(f"***detect_intent(): Title similarity of '{title}': {title_sim}")
                
            # If the similarity score of the title is above the threshold and higher than the currently highest one, save the courses index and the score as best fit
            if title_sim > title_threshold and title_sim > best_fit[1]:
                idx = get_past_idx(title)
                print(f"***detect_intent(): Found SIMILAR Title: {(title, title_sim)} -- Title by idx: {get_past_title(idx)}")
                best_fit = (idx, title_sim)
                
        # Return the best fitting title, if any was found with a similarity above the threshold (otherwise, best_fit is ('other', 0.0))
        if best_fit[1] > title_threshold:
            return intent_replies[detected_intent[0]], [best_fit]
        # Otherwise change the intent to free_description
        else: 
            return "", []
        
    elif detected_intent[0] == "feedback":
        c_feedback = give_feedback(user_input, last_recommendations)
        if len(c_feedback) > 0:
            chatbot_reply = "You gave the following feedback:\n" #### NOT SURE OB ICH DAS DRIN LASSE ODER NUR ZUM TESTEN
            for (c, f) in c_feedback:
                chatbot_reply += f"- {current_courses[c]['title']}: {f}\n"
                chatbot_reply += "\n"
            return chatbot_reply, c_feedback
        else:
            #### MAYBE: IF SIMILARITY ZU FEEDBACK ÜBER BESTIMMTEN THRESHOLD: NACHRICHT AUSGEBEN, DASS ERKANNT WURDE, DASS USER FEEDBACK GEBEN WILL, ABER NICHT ERKANNT WURDE, FÜR WELCHE(N) KURS(E) BZW SENTIMENT -> KURZE BESCHREIBUNG VON FEEDBACK-FORMAT & WEITER STATUS == FEEDBACK
            ############ & FALLS UNTER THRESHOLD: NÄCHST-HÖCHSTEN INTENT VERSUCHEN
            chatbot_reply = "I think you wanted to give feedback to one or more of the recommendated courses, but I could not clearly understand you. Please click on the button 'Feedback Hint' below the chat to find out how to properly give feedback."
            return chatbot_reply, []

            #detected_intent, intent_replies[detected_intent], []


def next_intent(intent_similarities, last_intent):
    """
    Finds the intent with next highest similarity to the input

    Parameters:
        intent_similarities (dict): dictionary containing the intents sorted by their similarity to the input
        last_intent (str): the last checked intent
    Returns:
        the next most similar intent after the last intent
    """
    # Get the list of intents (sorted by similarity)
    intents = list(intent_similarities.keys())
    
    # Get the index of the last intent
    index = intents.index(last_intent)
    
    # Check if there's a next entry
    if index + 1 < len(intents):
        next_intent = intents[index + 1]
        return next_intent
    else:
        return 'other'


def course_reference(user_input, user_embedding):
    """
    Tries to find the referenced course. If no fitting title is found, it calls the next highest intent
    """
    # First check if a title is spelled out exactly (only one reference per message allowed)
    best_fit = ("", 0.0)
    all_titles = [c['title'] for c in past_courses]
    for title in all_titles:
        ##### MAYBE EHER VARIANTE NUTZEN, DIE AUCH LEICHTE ABWANDLUNGEN (z.B. DURCH TIPPFEHLER) ERKENNT
        ####### DANN VLLT LISTE (ODER DICT MIT SIMILARITY-WERTEN) ERSTELLEN & DA ALLE TITEL, DIE ÄHNLICH GENUG SIND, DRIN SPEICHERN & AM ENDE BEST FIT RETURNEN
        ### IST DAS ÜBERHAUPT NOTWENDIG???
        if title in user_input:
            idx = get_past_idx(title)
            print(f"***detect_intent(): Found EXACT Title: {title} (by index {idx}: {past_courses[idx]['title']})")
            #detected_courses.append((title, 1.0))  ### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
            return detected_intent, intent_replies[detected_intent], (idx, 1.0)
        
    
    # Substract the difference between the similarity of liked course references and free description from the threshold  ### OR BETTER JUST SIM OF REFERENCE??
    # -> the more certain it is to be a reference, the lower is the threshold for the title similarity
    ### NICHT SICHER, OB DAS WIRKLICH GUT IST (WEGEN NONSENSE)
    #title_threshold = 0.55 - (intent_similarities['liked_course_reference'] - intent_similarities['free_description'])
    #print(f"***detect_intent(): Title threshold = {title_threshold}")
    title_threshold = 0.5

    # Check if the input is similar enough to any of the titles
    for title, title_emb in title_embeddings.items():  ### KANN ICH SONST AUCH ÜBER INDEX MACHEN -> BRAUCHE DANN TITLE NICHT MEHR IN TITLE_EMBEDDINGS -> IST DANN NUR NOCH LIST STATT DICT
        title_sim = cosine_similarity([user_embedding], [title_emb])
        #print(f"{title}: {title_sim}")
        # If a part of the title was detected in user input, set the threshold lower
        
        ## JUST TESTING 
        if title_sim > 0.3:
            print(f"***detect_intent(): Title similarity of '{title}': {title_sim}")
            
        # If the similarity score of the title is above the threshold and higher than the currently highest one, save the courses index and the score as best fit
        if title_sim > title_threshold and title_sim > best_fit[1]:
            idx = get_past_idx(title)
            print(f"***detect_intent(): Found SIMILAR Title: {(title, title_sim)} -- Title by idx: {get_past_title(idx)}")
            best_fit = (idx, title_sim)
            
    # Return the best fitting title, if any was found with a similarity above the threshold (otherwise, best_fit is ('other', 0.0))
    if best_fit[1] > title_threshold:
        return detected_intent, intent_replies[detected_intent], best_fit
    # Otherwise change the intent to free_description
    else:  ### MAYBE BESSER: VERSUCHE NÄCHST HÖCHSTEN INTENT?!? -> DAFÜR AUSFÜHRUNG DER INTENTS ALS EIGENE FUNKTIONEN (& Liste mit similarities übergeben; + default-intent, der aufgerufen wird, wenn Liste leer)?!?
        # Get the next highest intent
        
        
        detected_intent = "free_description"

###--- Handle Feedback ---###

def find_feedback_courses(sentence):
    """
    Find the index/indices of the course(s) the user gave feedback for

    Parameters:
        sentence (str): a sentence that might include a reference to a course
    Return:
        either list of indices (int) or list containing a string ('all'/'none')
    """
    #### ALSO CHECK FOR RANGES? (e.g., "I like the first three", "I like 2 to 4")
    # Sorting words to the corresponding positions
    c_position = [("last", "final"), ("1", "one", "first"), ("2", "two", "second"), ("3", "three", "third"), ("4", "four", "fourth"), ("5", "five", "fifth")]
    position_keys = {word: idx for idx, tpl in enumerate(c_position) for word in tpl}
    merge_numbers = {"second one": "2", "third one": "3", "fourth one": "4", "fifth one": "5", "last one": "0"}
    all_courses = {"all": "all", "any": "all", "every": "all", "everyone": "all", "none": "none", "no": "none"}
    
    # Make the sentence cases-insensitive and split it into separate words
    split_input = sentence.replace(",", " , ").lower().split()  # Add a space before each comma to count it as a word, separating numbers in an enumeration
    
    # First check if the user gave feedback for all courses simultaneously
    mentioned_all = [all_courses[word] for word in split_input if word in all_courses]
    mentioned_all = list(set(mentioned_all))  # Remove duplicates
    if len(mentioned_all) == 2:  # If the sentence includes both 'none' and 'all', interpret it as 'none' ###?!? ODER DANN BESSER IGNORIEREN???
        return ["none"]
    elif len(mentioned_all) == 1:
        return mentioned_all

    # Replace word pairs like "fourth one" with a single number (in this case: "4")
    idx_matches = []
    skip_next = False
    for idx, word in enumerate(split_input):
        # Skip the word if it was merged with the previous one
        if skip_next:
            skip_next = False
            continue
        if len(split_input) > idx + 1:
            next_merge = " ".join([word, split_input[idx+1]])
            if next_merge in list(merge_numbers):
                idx_matches.append(int(merge_numbers[next_merge])-1)
                skip_next = True
            elif word in position_keys:
                idx_matches.append(int(position_keys[word])-1)
        elif not skip_next and word in position_keys:
            idx_matches.append(int(position_keys[word])-1)
    # Remove duplicates
    idx_matches = list(set(idx_matches))
    return idx_matches

def sentence_sentiment(sentence):
    """
    Interpret if the user likes or dislikes the courses mentioned in the given sentence

    Parameters:
        sentence (str): sentence to check the sentiment of
    Returns:
        'liked', 'disliked' or 'negation', or None if no sentiment was found
    """
    sentiment_dict = {
        "liked": ["liked", "like", "love", "interesting", "good", "awesome", "nice", "great"],
        "disliked": ["dislike", "hate", "boring", "bad"],
        "negation": ["not", "don't", "didn't", "doesn't", "aren't", "isn't"]
    }
    # Reverse the dictionary for a more efficient lookup
    sentiment_key = {word: key for key, values in sentiment_dict.items() for word in values}

    # Split the sentence into words
    split_feedback = re.sub(r"[^\w\s']", '', sentence).lower().split()
    matches = list(set([sentiment_key[word] for word in split_feedback if word in sentiment_key]))

    # If exactly one sentiment is found, return it
    if len(matches) == 1 and matches[0]:
        return matches[0]
    # If only 'negation' and 'liked' are found (e.g., "I don't like course 1"), return 'disliked'
    # Not the other way around, because phrases like "I don't hate it" don't necessarily mean that it's liked
    elif len(matches) == 2 and 'negation' in matches and 'liked' in matches:
        return 'disliked'
    print(f"\n\n?!?sentence_sentiment(): Found {len(matches)} matches: {matches}\n\n")
    return None

def give_feedback(user_input, last_recommendations):
    """
    Processes and interprets the user's feedback

    Parameters:
        user_input (str): the user's feedback
        last_recommendations (list): list of the indices of the last recommended courses
    Return:
        List of tuples with course indices (from current_courses) and corresponding feedback 
    """
    # If only one course was recommended, directly check the sentiment
    if len(last_recommendations) == 1:
        print(f"***give_feedback(): Only one recommendation: {last_recommendations[0]}")
        sentiment = sentence_sentiment(user_input)
        if sentiment is None:
            return[]
        else:
            return [(last_recommendations[0], sentiment)]

    # #### BERÜCKSICHTIGEN, OB PUNKT SATZENDE IST ODER ZUR NUMMERIERUNG GEHÖRT (z.B. "I liked the 1. and 2. course")
    # Split the input into parts separated by 'but'
    parts = []
    if 'but' in user_input.lower():
        start = 0
        # Find all occurrences of 'but'
        while True:
            index = user_input.lower().find('but', start)
            if index == -1:
                break
            
            # Add the part before the occurence to the list and move the start position to the index after the occurence
            parts.append(user_input[start:index].strip())
            start = index + 3

        # Add the remaining part (after last 'but')
        parts.append(user_input[start:].strip())
    else:
        parts.append(user_input)
    # Split each part of the input by punctuation (.!?) to separate the sentences (if there are multiple)
    parts = [re.split(r'[.!?]', sent.strip()) for sent in parts]

    # Combine the parts (currently list of lists) to a one dimensional list and remove empty elements
    sentences = []
    for s in parts:
        sentences.extend(s)
    sentences = list(filter(None, sentences))
    
    # Find course references and sentiments for each sentence
    given_feedback = []
    #print("SENTENCES:", sentences)
    for s in sentences:
        courses = find_feedback_courses(s)
        # If no course was found, skip the sentence
        if len(courses) == 0:  
            continue
        
        c_sentiment = sentence_sentiment(s)

        # If no sentiment was found, skip the sentence
        if c_sentiment is None:
            continue

        # If the sentiment is 'negation', use the opposite of the previous sentence's sentiment (if any)
        if c_sentiment == 'negation' and len(given_feedback) > 0:
            print(f"***give_feedback(): Found feedback {c_sentiment} in sentence {s}!")
            c_sentiment = given_feedback[-1][1]
            print(f"*** -> Sentiment of last sentence: {given_feedback[-1][1]}")
            c_sentiment = 'liked' if given_feedback[-1][1] == 'disliked' else 'disliked'
            print(f"*** -> Sentiment of current sentence changed to {c_sentiment}")
        # If it is negation but there have not yet been sentences with a detected sentiment, skip the sentence
        elif c_sentiment == 'negation':  
            continue

        # If the user gave feedback for all courses simultaneously, set all courses to the sentiment
        # If the user stated that they like none of the recommended courses, set all to 'dislike'
        if courses[0] == 'all' or (courses[0] == 'none' and c_sentiment == 'liked'):
            #given_feedback = {c: c_sentiment if courses[0] == 'all' else 'dislike' for c in last_recommendations}
            given_feedback = [(c, c_sentiment) if courses[0] == 'all' else (c, 'dislike') for c in last_recommendations]
        # If the user specified the positions of the courses they (dis)liked, set each mentioned course separately to the sentiment
        elif isinstance(courses[0], int):
            for i in courses:
                if i >= len(last_recommendations):  ### EITHER TELL THE USER (MIGHT BE A TYPO?) OR JUST IGNORE IT THEN??
                    print(f"*-*-*give_feedback(): Index {i} is not in the recommendations?!")
                else:
                    given_feedback.append((last_recommendations[i], c_sentiment))
    print(f"\n***give_feedback(): given_feedback: {given_feedback}\n")
    return given_feedback


###--- User Preference Management ---### 
##### EVTL UMBENENNEN ZU update_user_preferences
def update_user_profile(user_profile, input_embedding = None, rated_course = None, liked=True, learning_rate=0.25):
    """
    Updates the user's preferences according to a free description, a reference to a liked course, or given feedback

    Parameters: 
        user_profile (array?): current embedding of the user's preferences
        input_embedding (array?)*: embedding of either the user's free description (transformed into a soup)
        rated_course ((int, str))*: either (index, 'past') of a previously liked course or (index, 'current') of a course the user gave feedback for
        liked (bool): if the user liked the course / the recommendation of the course; for descriptions always True
        learning_rate (float): how strong the feedback should influence the user's preferences
        * either input_embedding or rated_course is necessary
    Returns: 
        updated user_profile
    """
    print("***update_user_profile(): Updating user profile...")
    # If no input_embedding is given, set it to the embedding of the rated course
    if input_embedding is None:
        print(f"***update_user_profile(): Using: {rated_course}...")
        if rated_course[1] == 'past':
            #course_idx = [c['title'] for c in past_courses].index(rated_course[0]) ### ONLY NECESSARY, IF COURSES SAVED WITH TITLE INSTEAD OF IDX
            #input_embedding = past_embeddings[course_idx]
            input_embedding = past_embeddings[rated_course[0]]  ## IF COURSES SAVED BY IDX
        else:
            #course_idx = [c['title'] for c in current_courses].index(rated_course[0]) ### ONLY NECESSARY, IF COURSES SAVED WITH TITLE INSTEAD OF IDX
            #input_embedding = current_embeddings[course_idx]
            input_embedding = current_embeddings[rated_course[0]]  ## IF COURSES SAVED BY IDX
    if user_profile is not None:
        if liked:  # Positive feedback, liked course or free description
            user_profile += learning_rate * input_embedding  # Make the embedding of the user's preferences more similar to the input or liked course
        else:      # Negative feedback
            user_profile -= learning_rate * input_embedding  # Make the embedding of the user's preferences less similar to the disliked course
    else:
        # If no profile exists, set user's embedding to the input embedding
        user_profile = input_embedding
    return user_profile


###--- Filter Management ---###

####################################################
##### SETTING FILTERS FROM FREE DESCRIPTIONS #######
def find_sws_ects(user_input, old_filter):
    print("===============================\nChecking SWS and ECTS in input...")
    #sws_ects_pattern = r"(?:\b|\s|\.)(\d\ssws|\d\sects)"
    old_sws_ects_pattern = r"""
        (?:\b|\s\.)
        (
            (\d+)                           # First sws/ects of range
            (?:\sto\s|\-|\s\-\s|\sand\s)    # Range indicator
            (\d+\ssws|\d+\sects)            # Last sws/ects of range (with indicator)
        |
            (\d+\ssws|\d+\sects)            # Single mention
        )
        """
    sws_ects_pattern = r"""
        (?:\b|\s\.)
        (
            (\d+)                           # First sws/ects of range
            (?:\sto\s|\-|\s\-\s|\sand\s)    # Range indicator
            (\d+\s?sws|\d+\s?ects)          # Last sws/ects of range (with indicator)
        |
            (\d+\s?sws|\d+\s?ects)          # Single mention
        )
        """
    # Get all mentioned SWS and ECTS from the input
    matches = re.findall(sws_ects_pattern, user_input, re.VERBOSE | re.IGNORECASE)
    found_matches = [m[0] for m in matches]
    #print(f"-> SWS/ECTS mentioned in input: {found_matches}\n({matches})")
    cleaned_input = user_input

    found_sws = []
    found_ects = []
    
    for match in found_matches:
        #print(f"Checking {match}...")
        # Remove SWS and ECTS from input to avoid misinterpreting them as times
        cleaned_input = cleaned_input.replace(match, "")
        # Extract the attribute and number(s) from the match
        attr = re.search(r'(ects|sws)', match, flags = re.IGNORECASE).group().strip().lower()
        numbers = [int(nr) for nr in re.findall(r'\d+', match)]
        # Check for ranges
        if any(range_indicator in match for range_indicator in ['and', '-', 'to']):
            numbers = list(range(numbers[0], numbers[1]+1))
            #print(f"~~> Found range: {numbers}")
        if attr == 'sws':
            found_sws += numbers
        elif attr == 'ects':
            found_ects += numbers
        else:
            print(f"-x-x-x-x- {attr} is neither sws nor ects??!!")
        #print(f"~~> Found: {attr} -- {numbers}")

    # Combine them with previously set filters
    #found_sws += old_filter
    found_sws += old_filter['sws']
    found_ects += old_filter['ects']

    # Remove duplicates and sort the lists
    found_sws = sorted(list(set(found_sws)))
    found_ects = sorted(list(set(found_ects)))
    print(f"Found SWS: {found_sws}\nFound ECTS: {found_ects}\nCleaned input: {cleaned_input}")
    return found_sws, found_ects, cleaned_input

def input_times(user_input, old_filter):
    print("===============================\nChecking days and times in input...")
    #print("Checking days in input...")
    weekdays = {'monday': 'Mon.', 'tuesday': 'Tue.', 'wednesday': 'Wed.', 'thursday': 'Thu.', 'friday': 'Fri.', 'saturday': 'Sat.',
                'every day': 'alldays', 'each day': 'alldays', 'any day': 'alldays', 'all days': 'alldays', 'everyday': 'alldays', 'every single day': 'alldays', 
                'other\sday': 'otherdays', 'other\sdays': 'otherdays', 'remaining days': 'otherdays'}
    all_weekdays = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.']
    
    # Regex pattern to match weekdays (multiple days / ranges as well as individual days)
    #### KLAPPT DAS SO AUCH FÜR AUFLISTUNG VON TAGEN??? z.B. "On mondays, tuesdays and fridays..."
    abbrev_weekdays_pattern = r"""
        (?:\b|\s)
        (
            (from\s|between\s)? 
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.)          # First day of range
            (\sto\s|\-|\s\-\s|\sor\s|\sand\s)                              # Range indicator
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.)        # End time
        |
            (Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.|alldays|otherdays)         # Single day
        )
        """
    
    found_days = []  # List to store all found days
    found_day_time = {}  # days as keys, corresponding times as values

    # Replace full weekdays with abbreviations
    for full_day, abbrev in weekdays.items():
        user_input = re.sub(rf'\b{full_day}s?\b', abbrev, user_input, flags=re.IGNORECASE)
    #print(f"DAYS Changed input to: {user_input}\n")
    matches = re.findall(abbrev_weekdays_pattern, user_input, re.VERBOSE | re.IGNORECASE)
    found_days = [m[0] for m in matches]
    print(f"-> Days mentioned in input: {found_days}")

        
    ################################################################
    ##### T I M E S
    ################################################################

    # Get all times from input
    #print("\n-----\nChecking times in input...")

    # Add a mapping for textual numbers to digits
    textual_time_map = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", 
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", 
        "eleven": "11", "twelve": "12"
    }

    # Define regex pattern for times
    time_pattern = r"""
        \b
        (
            (?:at\s|from\s|between\s)?                                                              # Optional range indicator at start
            (half\spast\s|quarter\sto\s|quarter\spast\s)?                                           # Optional time modifiers
            ([1-9]|1[0-9]|2[0-3])(:[0-5][0-9])?                                                     # Starting time
            ((\s|\.|\b)?(?:o'clock\s)?(?:in\s)?(?:the\s)?(morning|afternoon|evening|AM|PM|am|pm)?)  # Optional modifiers
            (\sto\s|\sand\s|\sand\send\sat\s|\suntil\s|\-|\s\-\s)                                   # Range indicator
            (half\spast\s|quarter\sto\s|quarter\spast\s)?                                           # Optional time modifiers
            (([1-9]|1[0-2]|2[0-3])(:[0-5][0-9])?)                                                   # End time
            ((\s|\.|\b)?(?:o'clock\s)?(?:in\s)?(?:the\s)?(morning|afternoon|evening|AM|PM|am|pm)?)  # Optional modifiers
        |
            (at\s|half\spast\s|quarter\sto\s|quarter\spast\s)?  # Optional time indicators
            ([1-9]|1[0-9]|2[0-3])(:[0-5][0-9])?                 # Hours and optional minutes
            (\s|\.|\b)?                                         # Optional space, period, or word boundary
            (?:o'clock\s)?                                      # Optional "o'clock"
            (?:in\s)?                                           # Optional "in"
            (?:the\s)?                                          # Optional "the"
            (morning|afternoon|evening|AM|PM|am|pm)?            # Optional time of day
        )
        \b
        """
    
    ### AUCH GROBE ZEITABSCHNITTE WIE "morning / noon / afternoon / evening / all day" OHNE GENAUE ZEITANGABEN ZULASSEN??
    ### V.A. "all day / the whole day / anytime" WICHTIG I GUESS?? -> WIRD ABER EH ASSUMED, WENN KEINE ZEIT GEFUNDEN WIRD

    # Replace textual times with numbers
    for text_time, numeric_time in textual_time_map.items():
        user_input = re.sub(rf'\b{text_time}\b', numeric_time, user_input)
    #print(f"TIME Changed input to: {user_input}")

    ######################################################################################################
    #### SPLITTING THE INPUT BASED ON DAYS AND TIMES
    #print("\n---//---//---//---\n")
    # Split the input text at day or day range boundaries
    ## DOES NOT DETECT TIMES WITHOUT DAYS (should be interpreted as 'alldays') ### LASSE ICH ERSTMAL WEG
    split_pattern = "|".join(found_days)
    parts = re.split(rf"(?={split_pattern})", user_input, flags=re.IGNORECASE)
    cleaned_parts = [part for part in parts if any(day in part for day in found_days)]
    parts_dict = dict(zip(cleaned_parts, found_days))
    #print(f"PARTS DICT: {parts_dict}\n")

    def extract_times(input_part):
        found_phrases = []
        found_times = []
        matches = re.findall(time_pattern, input_part, re.VERBOSE)
        
        # Iterate through each match to clean up the results
        for match in matches:
            # match contains multiple capturing groups, we need to check which part was matched
            if match[0]:  # General time (e.g., "at 3 pm")
                found_phrases.append(match[0].strip())
            elif match[1]:  # Time range: "from X to Y"
                found_phrases.append(match[1].strip())
            elif match[2]:  # Time range: "between X and Y"
                found_phrases.append(match[2].strip())
        #print(f"Found phrases: {found_phrases}")

        # If no times were found: Return whole day (0:00 - 24:00)
        if len(found_phrases) == 0:
            return [[0, 24]]

        # For each found time, check for modifiers
        for time in found_phrases:
            #print(f"\n-:-:-:-:- Checking '{time}'...")
            numbers = re.findall(r'(\d+\:\d+|\d+)', time)
            #print(f"xx Found '{numbers}'")
            if len(numbers) == 1:
                number = re.search(r'\d+', numbers[0])
                hour = int(number.group()) # Convert matched number to integer
                #print(f"x+x+ Found '{hour}'")
                if bool(re.search(r'(\b|\d+)(pm|afternoon|evening)\b', time.lower())) and hour <= 12:
                    #print(f"Increasing {hour}")
                    hour += 12
                found_times.append([hour, hour+2])
                #print(f"->-> Appended {found_times[-1]}")
                    
            elif len(numbers) == 2:
                start_number = re.search(r'\d+', numbers[0])
                end_number = re.search(r'\d+', numbers[1])
                start_hour = int(start_number.group()) # Convert matched start number to integer
                end_hour = int(end_number.group()) # Convert matched end number to integer
                has_am = bool(re.search(r'(\b|\d+)(am|morning)\b', time.lower()))
                has_pm = bool(re.search(r'(\b|\d+)(pm|afternoon|evening)\b', time.lower()))
                #print(f"Found '{(start_hour, end_hour)}'\nAM: {has_am} -- PM: {has_pm}")

                # If both am and pm are found or end is smaller than start, end is pm
                if ((has_am and has_pm) or (end_hour < start_hour)) and end_hour <= 12:
                    #print(f"Increasing end '{end_hour}'")
                    end_hour += 12

                # If only pm is found and end_hour is bigger than the start, it is for both
                elif has_pm and end_hour > start_hour and end_hour <= 12:
                    #print(f"Increasing both '{start_hour}' & '{end_hour}'")
                    start_hour += 12
                    end_hour += 12
                found_times.append([start_hour, end_hour])
                #print(f"->-> Appended '{found_times[-1]}'")
            else:
                print(f"\n\n-!-!-!- WHAT IS THIS SHIT?!?! -- '{numbers}'\n\n")
        #print(f"x+x+x extract_times() returns {found_times}")
        return found_times
    

    def add_time(days, times):
        """
        Parameters:
            day (list): list of all days the times should be added to
            times (list): list of times to add to all given days
        """
        #print(f"-x-x-x- Adding '{times}' to '{days}'")
        if not isinstance(times, list):
            times = [times]
            #print(f"--- CHANGED TYPE OF TIMES TO: {type(times)}")
        if not isinstance(days, list):
            days = [days]
            #print(f"--- CHANGED TYPE OF DAYS TO: {type(days)}")
        for day in days:
            for time in times:
                if day in found_day_time:
                    found_day_time[day].append(time)
                else:
                    found_day_time[day] = [time]

    for part, days in parts_dict.items():
        print(f"-=-=-=-=-=-=-=-=-=-=-=-=-\nChecking part '{part}' (day '{days}')...")
        # Get the list of times for the current part of the input
        times = extract_times(part)

        # Check if 'alldays' is given -> set times for all days of the week
        if days == 'alldays':
            print(f"-> Found all days!\n{times}")
            #for weekday in all_weekdays:
            add_time(all_weekdays, times)
            #print(f"->->-> SET TIMES: {found_day_time}")
        
        # Check if 'otherdays' is given -> set times for all days that have no time yet
        elif days == 'otherdays':
            #print(f"-> Found other days!")
            other_days = [d for d in all_weekdays if not (d in found_day_time.keys())]
            print(f"~~> Other days: {other_days}!\n{times}")
            add_time(other_days, times)

        # Check if a range of days is given
        elif any(range_indicator in days for range_indicator in ['between', '-', 'to']):
            #print(f"-> Found a range!")
            found_day_range = []
            for range_day in all_weekdays:
                if range_day in days:
                    #print(f"~~ Found start/end of range: {range_day}")
                    found_day_range.append(range_day)
                    # If other days have already been appended, this was the end of the range
                    if len(found_day_range) > 1:
                        break
                # If this day is not in this part of the input, only append it if the first day of the range was already appended
                elif len(found_day_range) > 0:
                    #print(f"~~ Appending '{range_day}' to range")
                    found_day_range.append(range_day)
            print(f"~~> Full range: {found_day_range}\n{times}")
            add_time(found_day_range, times)

        # Otherwise, set times for the individual days in the current part of the input
        else:
            #print(f"-> Found individual days!")
            found_individual_days = []
            for ind_day in all_weekdays:
                if ind_day in days:
                    print(f"~~ Found: {ind_day}\n{times}")
                    found_individual_days.append(ind_day)
            add_time(found_individual_days, times)
            #print(f"->->-> Added time: {found_day_time[found_individual_days[0]]}")
    #print(f"\n\n==================================\nTime dict: {found_day_time}")

    # Add the new times to the previously set times
    if old_filter == []:
        old_filter = {}
    
    for found_day, found_times in found_day_time.items():
        if found_day in old_filter:
            old_filter[found_day] += found_times
        else:
            old_filter[found_day] = found_times

    # Merge overlapping timeframes
    merged_times = {}
    for found_day, found_times in old_filter.items():
        found_times.sort(key=lambda x: x[0])

        # Initialize the merged list with the first interval
        merged = [found_times[0].copy()]
        
        for current in found_times[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current[0] <= previous[1]:
                #print(f"FOUND OVERLAP: {current[0]} <= {previous[1]}")
                # Merge the intervals
                previous[1] = max(previous[1], current[1])
            else:
                # No overlap, add the interval to the result
                merged.append(current)
        merged_times[found_day] = merged

    # Sort merged times by days
    merged_times = {day: merged_times[day] for day in all_weekdays if day in merged_times}
    return merged_times

def find_modules(user_input, old_filter):
    """
    Find modules in given input

    Parameter:
        user_input (str): the user's input in which modules should be found
        old_filter (list): list of modules currently filtered for
    Returns:
        list of all found modules

    """
    #print(f"\n\n***find_modules(): START FINDING MODULES")
    # Module names are constructed from 
    # > "CS-" for Cognitive Science (dataset includes only Cognitive Science courses, therefore all modules start with "CS-")
    # > + "[B/M][WP/W/P]-" -- "B" if Bachelor, "M" if Master; "WP" if elective, "P" if compulsory, "W" if "Distinguishing Elective Courses"
    # > + Short form of the area (e.g. "AI")
    all_modules = list(set([m.split(" > ")[0].split(",")[0] for m in current_attributes['module']]))
    all_modules.sort()
    modules = []
    found_area = None
    found_program = []
    found_module = []

    # First look for an area - if that is not given, return an empty list
    # Split the input into words and remove punctuation
    split_input = user_input.lower().split()
    split_input = [word.translate(str.maketrans('', '', string.punctuation)) for word in split_input]
    #print(f"***find_modules(): split_input: {split_input}")
    for key, area in module_dict['area'].items():
        #print(f"\n***find_modules(): checking ({key}, {area})")
        split_key = key.split()
        if len (split_key) == 1:  # if key is a single word, search it in split input
            if key in split_input:
                found_area = area
                #print(f"***find_modules(): Found area: {found_area} (from input '{key}')")
                break # Allow only 1 module per message   #### TELL USER!!!
        else:  # if key consist of multiple words, search for it in the input sentence(s)
            if key in user_input.lower():
                found_area = area
                #print(f"***find_modules(): Found area: {found_area} (from input '{key}')")
                break

    if found_area is None:
        return []
    for p in module_dict['study_program']:
        if p in user_input.lower():
            found_program = [module_dict['study_program'][p]]
            #print(f"***find_modules(): Found study_program: {module_dict['study_program'][p]} (from input '{p}')")
            break
    # If found area is empty, it is Distinguishing Elective Courses, which is module "W"
    if found_area == "":
        found_module = "W"
    else:
        for m in module_dict['module']:
            if m in user_input.lower():
                # If 'compulsory' is found, check if it is actually non compulsory (i.e., elective)
                if m == 'compulsory' and bool(re.search(rf'\b(non compulsory|non-compulsory)\b', user_input.lower())):
                    found_module = ['WP']
                else:
                    #print(f"***find_modules(): Found module: {module_dict['module'][m]} (from input '{m}')")
                    #found_module.append(module_dict['module'][m])
                    found_module = [module_dict['module'][m]]
                break
    #print(f"***find_modules(): found_program: {found_program} -- found_module: {found_module}")

    # If only the area was found and it is in the old filter, return only the old filter (in case the user just mentioned, e.g., AI without wanting to set the modules to each module that has '-AI')
    prog_found = True if len(found_program) > 0 else False
    mod_found = True if len(found_module) > 0 else False
    for old_f in old_filter:
        #print(f"Old Module: {old_f}")
        if area in old_f:
            if not prog_found and not mod_found:
                #print(f"Module {area} is already in filter_dict!")
                return old_filter
            # If only the study program or module type is not found, set it to the one(s) from the old filter
            old_program = re.findall(r'CS\-(B|M)', old_f)[0]
            if not old_program in found_program:
                #print(f"***find_modules(): Found old study program: {old_program}")
                found_program += old_program
            old_mod = re.findall(r'CS\-.([A-Z]+)\-', old_f)[0]
            if not old_mod in found_module:
                #print(f"***find_modules(): Found old mod: {old_mod}")
                found_module += old_mod

    # If the program or module is still empty, append each possible value
    if len(found_program) == 0:
        found_program = list(module_dict['study_program'].values())
    if len(found_module) == 0:
        found_module = list(module_dict['module'].values())


    # Combine all found parts and append and return all existing modules
    for p in found_program:
        for m in found_module:
            module = f"CS-{p}{m}{found_area} - "
            #print(f"***find_modules(): module in for-loop: {module}")
            modules += [mod for mod in all_modules if module in mod]
    #print(f"***find_modules(): found modules: {modules}")
    return modules

def find_attributes(user_input, old_filter_dict):
    """
    Extracts all attributes from the input

    Parameters:
        user_input (str): the user's input that should be checked for attributes
    Returns:
        dictionary with all found attributes (str) and their values (lists)
    """
    relevant_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'filter_time']

    # Get a dictionary containing all possible values for each attribute that is relevant for the soup
    check_attr = {a: v for a, v in current_attributes.items() if a in relevant_attributes} ### VON current_attr (da nur mit current_courses verglichen) ODER BESSER BEIDEN (da es auch Werte geben kann, die es in früheren Semestern gab, aber jetzt nicht mehr)???
    found_attr = {key: [] for key in check_attr}
    #print(f"Checking attributes: {check_attr}")

    for attr, val in check_attr.items():
        old_filter = old_filter_dict[attr] if attr in old_filter_dict else []
        print(f"***find_attributes(): checking attr '{attr}\n---- Old filter: {old_filter}'")
        # First check for individually processed attributes
        if attr == 'module':
            #found_attr[attr] = str(find_modules(user_input))
            found_attr[attr] = find_modules(user_input, old_filter)
            #print(f"")
            continue
        elif attr == 'filter_time':
            found_attr[attr] = input_times(user_input, old_filter)
            #print(f"")
            continue
        # int for SWS or ECTS points have to be directly followed by 'SWS' or 'ECTS' (except for ranges -> only second int has to be followed by it)
        elif attr in ['sws', 'ects']:
            #if 'sws' in user_input.lower() or 'ects' in user_input.lower():
            # Only check for sws/ects if not checked before and at least one of the words is in the input
            if bool(re.search(r'\b(sws|ects)\b', user_input.lower())) and found_attr['sws'] == [] and found_attr['ects'] == []:
                old_sws_ects_filter = {sws_ects: old_filter_dict[sws_ects] if sws_ects in old_filter_dict else [] for sws_ects in ['ects', 'sws']}
                found_attr['sws'], found_attr['ects'], user_input = find_sws_ects(user_input, old_sws_ects_filter)
            continue
        elif attr == 'mode':
            found_modes = []
            # Check if one or multiple modes are given in the input (assuming that each mentioned mode is matching the format defined in the courses)
            matches = re.findall(r'\b((?:in person|hybrid|online)(\s(\+|with)\srecording)?)', user_input.lower(), re.IGNORECASE)  # No '\b' at the end of the regex pattern as there might also be a 's' after 'recording' and it is highly unlikely that 'recording' is just the start of a bigger word (in that case it is most likely that the user forgot a space anyway)
            found_matches = [m[0].replace("with", "+") for m in matches]
            print(f"matches: {found_matches}")
            # Check if 'recording' was found
            found_rec = True if 'recording' in " ".join(found_matches) else False
            # If 'recording' is mentioned in the input but not found in matches, append it to each found mode
            if (not found_rec) and bool(re.search(r'\b(recording)', user_input.lower(), re.IGNORECASE)): 
                print(f"Found 'recording' in input!")
                if found_matches:
                    print("Appending 'recording' to matches...")
                    found_modes = [m + ' + recording' for m in found_matches]
                else:
                    print("No matches, appending all modes with 'recording'...")
                    found_modes = ['online + recording', 'hybrid + recording', 'in person + recording']
            else:
                print("No need to add recording!")
                found_modes = found_matches
            found_modes += old_filter
            found_attr['mode'] = list(set(found_modes))
            print(f"FOUND MODES: {found_modes}")
            continue
        elif attr == 'status':
            found_status = []
            # Check if a status (or multiple) is given in the input (assuming that each mentioned status is matching the format defined in the courses)
            for status in current_attributes['status']:
                if status.lower() in user_input.lower():
                    print(f"Found status: {status}")
                    found_status.append(status)
                    # For lecture or seminar, also append 'Lecture and Practice' or 'Seminar and Practice' as there is not a huge difference; if a user does not want one with practice, they can delete the filter later
                    if status.lower() in ['lecture', 'seminar'] and not (status in old_filter):  # Don't append it if only 'Lecture' or 'Seminar' is in the previous filters, as that means that the user deleted the filter for 'Lecture and Practice' or 'Seminar and Parctice'
                        found_status.append((status + ' and Practice'))
            found_status += old_filter
            found_attr['status'] = list(set(found_status))
            continue

        # Check for each other attribute if a value is found
        found_attr[attr] += old_filter
        for v in val:
            # Add the value if it is found in the input and not yet in the list of found attributes or in the previous filter
            if str(v).lower() in user_input.lower() and str(v) != '' and not (v in found_attr[attr]) and not (v in old_filter):
                if isinstance(v, str):
                    found_attr[attr].append(v.title())
                else:
                    found_attr[attr].append(v)
    return found_attr

def set_filters(user_input, old_filter_dict):
    """
    
    
    """
    #filter_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'filter_time']
    found_attr = find_attributes(user_input, old_filter_dict)
    filter_dict = {attr: val for attr, val in found_attr.items() if val} ### STATTDESSEN BESSER UPDATEN (vorheriges dict als Parameter übergeben & neue Werte hinzufügen)
    print(f"***set_filters(): filter_dict: {filter_dict}")
    return filter_dict


###############################################
##### CHECKING IF COURSES MATCH FILTERS #######
def check_filter(filter_dict, course):
    ## Checks a single course
    missing_filters = 0  # Counts how many filtered attributes are missing (no values) for a course -> 
    #matching_filters = 0  # IDEE: Counts how many filters are matching for a course -> if no courses match all, maybe select courses with most matches? 
    for filter_key, filter in filter_dict.items():
        print(f"Checking filter '{filter_key}': '{filter}' (type: {type(filter)})")
        if not filter:
            continue
        # Check if the course has a value for the checked filter
        if (not course[filter_key]) or (re.search(r'(not specified)', str(course[filter_key]))):
            print(f"xXxXx Attribute '{filter_key}' is missing!!")
            missing_filters += 1
            continue
        
        # Check if every time of the course is in the filtered times
        if filter_key == 'filter_time':
            print("-> filter times...")
            for c_day, c_times in course['filter_time'].items():

                # Check if the day of the course is in the filtered times
                if (not (c_day in filter)) or (len(filter[c_day]) == 0):
                    print(f"-x-x-x-x- '{c_day}' is not in '{filter}'!!")
                    return False, missing_filters
                    
                # Check for each time of the day if it matches the filter
                #found_c_times = True ## If all times of the day were found in the filtered times ### PROP UNNÖTIG, DA RETURN FALSE
                for c_time in c_times:
                    # Times in filters are ordered -> checking smallest time first
                    found_time = False
                    for f_time in filter[c_day]:
                        # If the start time of the course is smaller than the start time of the filter or it matches the end time of the filter, it does not fit
                        #if (c_time[0] < f_time[0]) or (c_time[0] == f_time[1]):  ### UNNÖTIG WEGEN FOUND_TIME?!?
                        #    return False
                        # If the time of the course is within a timeframe of the filtered times for that day, there is no need to check for more filtered times in that day
                        if (c_time[0] >= f_time[0]) and (c_time[1] <= f_time[1]):
                            print(f"~~> Found matching time for day '{c_day}': {c_time}")
                            found_time = True
                            break
                    # If all filtered times for the day were checked but no fitting time was found, return False
                    if not found_time:
                        return False, missing_filters
                    
        # Filter for lecturer is interpreted as wanting at least one of those in the list (if there are more than one) -> one match is enough
        # 
        elif filter_key in ['lecturer_short', 'module']:
            found_filter = False
            print(f"----- Lecturer/Module: {course[filter_key]}")
            for c_val in course[filter_key]:
                if (c_val in filter):
                    print(f"~+~+~+~ Found {filter_key}: {c_val} in filter ({filter})")
                    found_filter = True
                    break
            if not found_filter:
                print(f"-x-x-x-x- None of these ({course[filter_key]}) are in filter!!")
                return False, missing_filters
        #elif filter_key == 'lecturer_short':
        #    found_lecturer = False
        #    print(f"----- Lecturer: {course[filter_key]}")
        #    for c_val in course[filter_key]:
        #        if (c_val in filter):
        #            print(f"~+~+~+~ Found {filter_key}: {c_val} in filter ({filter})")
        #            found_lecturer = True
        #    if not found_lecturer:
        #        print(f"-x-x-x-x- None of the lecturers ({course[filter_key]}) is in filter!!")
        #        return False, missing_filters
            
        # For some courses, the language is "German/English" -> both German and English must be in the filter
        elif filter_key == 'language':
            split_lang = course[filter_key].split('/')
            for lang in split_lang:
                if not (lang in filter):
                    print(f"-x-x-x-x- {filter_key}: {lang} is not in filter!!")
                    return False, missing_filters
                else:
                    print(f"~+~+~+~ Found {filter_key}: {lang} in filter ({filter})")


                    

        
        # All other filter attributes are stored in lists -> Just check if each value from the course matches the filter
        else:
            #for attr_val in filter:
            #print(f"Checking filter '{filter_key}': '{filter}'")

            if isinstance(course[filter_key], list):
                print(f"----- '{filter_key}' from course is a list! -> {course[filter_key]}")
                for c_val in course[filter_key]:
                    """for f_val in filter:
                        #print(f"F_VAL: {f_val}")
                        str_f_val = str(f_val)
                        if not bool(re.search(rf'\b{str(c_val)}\b', str_f_val)):
                            print(f"-x-x-x-x- {filter_key}: {c_val} is not in filter!!")
                        #if not (c_val in filter):
                            # Check if it is trying to compare str to int
                            if (str(c_val).isdigit() and not (int(c_val) in filter)) or not str(c_val).isdigit():
                                print(f"-x-x-x-x- {filter_key}: {c_val} is not in filter!!")
                                return False, missing_filters
                        else:
                            print(f"~+~+~+~ Found {filter_key}: {c_val} in filter ({filter})")"""
                    if not (c_val in filter):
                        print(f"-x-x-x-x- {filter_key}: {course[filter_key]} is not in filter!!")
                        return False, missing_filters
                    else:
                        print(f"~+~+~+~ Found {filter_key}: {course[filter_key]} in filter ({filter})")

            else:
                print(f"----- '{filter_key}' from course is a string! -> {course[filter_key]}")
                #if not (course[filter_key].lower() in [f.lower() for f in filter]):
                # Check if both values are of the same type
                if not isinstance(course[filter_key], type(filter[0])):
                    # Check if one of the values is an int   #### EIGTL MÜSSTEN ATTRIBUTE IN KURSEN IMMER STRINGS ODER LISTEN SEIN -> MUSS EIGTL NUR FILTER CHECKEN
                    if isinstance(course[filter_key], int) or isinstance(filter[0], int):
                        # Try to convert other to int -> if that's not possible, the filter cannot match the course's attribute
                        try:
                            course[filter_key] = int(course[filter_key])
                            filter = [int(f) for f in filter]
                        except:
                            print(f"Either {course[filter_key]} ({type(course[filter_key])}) or {filter} (type of elements: {type(filter[0])}) cannot be converted to int!!!")
                            return False, missing_filters
                        # If both are ints, compare them
                        if not (course[filter_key] in filter):
                            print(f"-x-x-x-x- {filter_key}: {course[filter_key]} is not in filter!!")
                            return False, missing_filters
                        else:
                            print(f"~+~+~+~ Found {filter_key}: {course[filter_key]} in filter ({filter})")

                        
                #if (course[filter_key])
                elif not (course[filter_key].lower() in [f.lower() for f in filter]):  ### MUSS NICHT PRÜFEN, OB BEIDE STR, DA ATTR IN KURSEN IMMER LIST (OBEN ABGEFANGEN) ODER STR SIND -> FALLS FILTER NICHT STR: OBEN ABGEFANGEN
                    print(f"-x-x-x-x- {filter_key}: {course[filter_key]} is not in filter!!")
                    return False, missing_filters
                else:
                    print(f"~+~+~+~ Found {filter_key}: {course[filter_key]} in filter ({filter})")
            """else:
                print(f"X=X=X=X DIFFERENT TYPE: '{filter_key}' from course is a {type(course[filter_key])}! -> {course[filter_key]}")
                for f_val in filter:
                    #print(f"F_VAL: {f_val}")
                    str_f_val = str(f_val)
                    if not bool(re.search(rf'\b{str(course[filter_key])}\b', str_f_val)):
                        print(f"-x-x-x-x- {filter_key}: {course[filter_key]} is not in filter!!")
                    #if not (course[filter_key] in filter):
                        # Check if it is trying to compare str to int
                        if (str(course[filter_key]).isdigit() and not (int(course[filter_key]) in filter)) or not str(course[filter_key]).isdigit():
                            print(f"-x-x-x-x- {filter_key}: {course[filter_key]} is not in filter!!")
                            return False, missing_filters
                    else:
                        print(f"~+~+~+~ Found {filter_key}: {course[filter_key]} in filter ({filter})")"""
                


                       
    # If more than 50% of filtered attributes are missing, return False
    if (missing_filters / len(filter_dict)) > 0.5:
        print(f"xXxXx '{course['title']}' is missing {missing_filters}/{len(filter_dict)} attributes!! -> Removed")
        return False, missing_filters
    
    # Otherwise, return True as each filter matched the course (otherwise, it would have returned at some point in the for-loop)
    else:
        return True, missing_filters


def filter_courses(filter_dict, courses):
    print("Start filtering...")
    # If no filters are set, return all current courses
    if len(filter_dict) == 0:
        print("No Filters Found!")
        return courses

    matching_courses = []  # Keys: courses; values: percentage (0-1) of how many filter attributes are missing in the course

    # Loop through every course and every filter
    for course in courses:
        print(f"\n-=-=-=-=-=-=- Checking {course['title']}...")
        is_matching, missing_filters = check_filter(filter_dict, course)
        if is_matching:
            print(f"~o~o~ Found matching course '{course['title']}' (missing {missing_filters} filters)")
            missing = (missing_filters / len(filter_dict))
            matching_courses.append([course, missing])
    #print(f"MATCHING: {matching_courses}")
    return matching_courses



###--- Recommendation Generation ---###

def input_embedding(user_input, filter_dict):
    """
    Encodes a given string

        Parameters:
            user_input (str): the user's input
            filter_dict (dict): currently set filters
        Returns:
            input_emb: the embedding of the input
            updated filter_dict
    """
    #relevant_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'time']
    updated_filter_dict = find_attributes(user_input, filter_dict)
    print(f"\n***~~~input_embedding(): Updated FILTER: {updated_filter_dict}")

    # Get the attributes to set filters for
    #filter_dict = {attr: val for attr, val in found_attr.items() if val != ""}
    #print(f"***set_filters(): filter_dict: {filter_dict}")

    # Get input in soup-format
    soup_dict = {attr: updated_filter_dict[attr].copy() if attr in updated_filter_dict else "" for attr in ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area']}
    soup = f"Title: . Description: {user_input}. Status: {soup_dict['status']}. Mode: {soup_dict['mode']}. ECTS: {soup_dict['ects']}. SWS: {soup_dict['sws']}. Lecturer: {soup_dict['lecturer_short']}. Module: {soup_dict['module']}. Area: {soup_dict['area']}."
    #print(f"\n***input_embedding(): FINAL soup_dict: {soup_dict}\n***input_embedding(): Soup: '{soup}'")

    #soup = input_soup(user_input)
    input_emb = model.encode([soup])[0]
    return input_emb, updated_filter_dict


def recommend_courses(user_profile, rated_courses, previously_liked_courses, filter_dict, amount=5):
    """
    Generates recommendations based on a user's preferences.

        Parameters: 
            user_profile (array?): embedding of the user's preferences
            rated_courses (list): indices of courses (in current_courses) the user has already rated
            previously_liked_courses (list): courses the user has stated to have liked (from past semesters)
            amount (int): how many courses should be recommended - default: top 5
            filter (dict): all selected filters
        Returns: 
            response: chatbot message before showing the recommendations
            response_end: chatbot message after showing the recommendations
            to_recommend: indices of the recommended courses
    """
    # Compute cosine similarity between the user profile and course embeddings
    similarities = cosine_similarity([user_profile], current_embeddings)[0]

    # Rank courses by similarity
    top_courses_indices = similarities.argsort()[::-1]
    print(f"***recommend_courses(): ~-~-~ Courses with top indices: {[current_courses[idx]['title'] for idx in top_courses_indices[:6]]}")

    #### MAYBE DELETE ALL COURSES WITH SIMILARITY BELOW THRESHOLD ALREADY HERE
    #### THEN CHECK IF LIST OF POSSIBLE RECOMMENDATIONS IS EMPTY: IF YES, SKIP REST AND WRITE RESPONSE
  
    # Delete already rated courses from top recommendations and select the specified amount of recommendations
    top_indices = [int(idx) for idx in top_courses_indices if idx not in rated_courses]
    print(f"\n***recommend_courses(): ~-~-~ Courses without ranted ones: {[current_courses[idx]['title']  for idx in top_indices[:6]]}")

    # Get the titles of previously liked courses
    liked_titles = [past_courses[idx]['title'] for idx in previously_liked_courses]

    # Delete all titles that are already in the previously liked courses
    cleaned_indices = [idx for idx in top_indices if current_courses[idx]['title'] not in liked_titles]
    print(f"\n***recommend_courses(): ~-~-~ Courses without prev. liked: {[current_courses[idx]['title']  for idx in cleaned_indices[:6]]}")

    response = ""
    response_end = ""
    to_recommend = []

    # If there are no more courses left that could be recommended, tell the user
    print(f"\n***recommend_courses(): len(cleaned_indices): {len(cleaned_indices)}")
    if len(cleaned_indices) == 0:
        response = f"There are no more courses left that I could recommend to you!"
        response_end = ""
        to_recommend = []
        
    # If there are less then the specified amount of courses left to recommend, tell the user that these are the last courses they have not yet rated or mentioned to have previously liked
    elif len(cleaned_indices) < amount:
        response = f"There are only {len(cleaned_indices)} courses left that I could recommend to you. These are:  \n"
        response_end = f"You already rated all the other courses or have taken them in previous semesters! I hope I was able to help you finding new courses and that I will see you again in a few months. Have a great semester!"
        to_recommend = cleaned_indices

    else:          
        # Delete courses that do not match the current filters
        cleaned_courses = [current_courses[idx] for idx in cleaned_indices]
        filtered_courses = filter_courses(filter_dict, cleaned_courses)
        filtered_indices = [current_courses.index(c) for c in filtered_courses]
        print(f"\n***recommend_courses(): ~-~-~ Courses filtered: {[c['title'] for c in filtered_courses]}")
        threshold = 0.7

        # If there are no courses left that match the current filters, ask the user to remove some filters
        if len(filtered_courses) == 0:
            response = f"There are no courses that match the currently set filters that you haven't rated or mentioned before! Please remove some filters by clicking on them in the list on the left side of the screen."
            response_end = ""
            to_recommend = []
            return response, response_end, to_recommend
            
        # If there are less then the specified amount of courses left to recommend, tell the user that these are the last courses they have not yet rated or mentioned to have previously liked
        elif len(filtered_courses) <= amount:
            response = f"There are only {len(filtered_courses)} courses that match the currently set filters that you haven't rated or mentioned before! These are:  \n"
            response_end = f"Please remove some filters by clicking on them in the list on the left side of the screen."
            to_recommend = filtered_indices
            return response, response_end, to_recommend

        # Decrease threshold if filters are set
        elif len(filtered_courses) < len(cleaned_indices):
            threshold = 0.5

        # Check if the similarity of any of the courses is above the threshold and select the corresponding response to return together with the list of courses to recommend
        print(f"***recommend_courses(): Best fits: {[(get_current_title(c), float(similarities[c])) for c in filtered_indices[:amount]]}")
        for course in filtered_indices[:amount]:
            if float(similarities[course]) >= threshold:
                to_recommend.append(course)
        if len(to_recommend) > 1:
            response = "I found some courses you might like:  \n"
            response_end = f"To get more information about a course, you can click on its title in the list.  \n\nPlease tell me if these courses sound interesting to you.  \nIf you haven’t done that already, please check out the ‘Feedback Hints’ (click on the button below the chat) to find out how to properly give feedback. "
        elif len(to_recommend) == 1:
            response = f"I found a course you might like:"
            response_end = f"Feedback would help a lot to improve further recommendations. Please tell me if this course sounds interesting or not. "
        else:
            response = "I need some more information to generate good recommendations for you. Could you tell me more about what kind of course you are looking for? Or is there any course you liked in the past that you didn't tell me about yet? "
        print(f"***recommend_courses(): Recommending: {[(get_current_title(c), float(similarities[c])) for c in to_recommend]}")

    return response, response_end, to_recommend