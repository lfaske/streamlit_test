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

# Define the intent detection function
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

def find_modules(user_input):
    """
    Find modules in given input

    Parameter:
        user_input (str): the user's input in which modules should be found
    Returns:
        list of all found modules

    """
    print(f"\n\n***find_modules(): START FINDING MODULES")
    # Module names are constructed from 
    # > "CS-" for Cognitive Science (dataset includes only Cognitive Science courses, therefore all modules start with "CS-")
    # > + "[B/M][WP/W/P]-" -- "B" if Bachelor, "M" if Master; "WP" if elective, "P" if compulsory, "W" if "Distinguishing Elective Courses"
    # > + Short form of the area (e.g. "AI")
    all_modules = list(set([m.split(" > ")[0].split(",")[0] for m in current_attributes['module']]))
    all_modules.sort()
    modules = []
    found_area = None
    found_program = list(module_dict['study_program'].values())
    found_module = list(module_dict['module'].values())

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
                #print(f"***find_modules(): Found module: {module_dict['module'][m]} (from input '{m}')")
                #found_module.append(module_dict['module'][m])
                found_module = [module_dict['module'][m]]
                break
    #print(f"***find_modules(): found_program: {found_program} -- found_module: {found_module}")

    # Combine all found parts and append and return all existing modules
    for p in found_program:
        for m in found_module:
            module = f"CS-{p}{m}{found_area} - "
            #print(f"***find_modules(): module in for-loop: {module}")
            modules += [mod for mod in all_modules if module in mod]
    print(f"***find_modules(): found modules: {modules}")
    return modules
    
def find_attributes(user_input, relevant_attributes):
    """
    Extracts all attributes from the input

    Parameters:
        user_input (str): the user's input that should be checked for attributes
    Returns:
        dictionary with all found attributes and their values
    """
    #relevant_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'time']

    # Get a dictionary containing all possible values for each attribute that is relevant for the soup
    check_attr = {a: v for a, v in current_attributes.items() if a in relevant_attributes} ### VON current_attr (da nur mit current_courses verglichen) ODER BESSER BEIDEN (da es auch Werte geben kann, die es in früheren Semestern gab, aber jetzt nicht mehr)???
    found_attr = {key: "" for key in check_attr}

    for attr, val in check_attr.items():
        #print(f"***input_soup(): checking attr '{attr}'")
        if attr == 'module':
            #found_attr[attr] = str(find_modules(user_input))
            found_attr[attr] = find_modules(user_input)
            #print(f"")
            continue
        # Check if a value is found
        for v in val:
            #print(f"***input_soup(): -- checking value '{v}'")

            if str(v).lower() in user_input.lower() and str(v) != '':
                attr_key = attr

                # If the value appears in multiple keys (attributes), check if one of the attributes is stated in the input
                all_keys = [key for key, values in check_attr.items() if str(v) in values]
                #print(f"ALL KEYS: {all_keys}")
                #print(f"\n***input_soup(): Found VALUE '{v}' (for attr '{attr}') in input! -> exists in {len(all_keys)} keys!")

                if len(all_keys) > 1:
                    input_keys = [key for key in all_keys if key in user_input.lower()]
                    #print(f"***input_soup(): ALL KEYS: {all_keys} -- INPUT KEYS: {input_keys}")
                    if len(input_keys) > 1:
                        #print(f"***input_soup(): selecting closest key...")
                        # Select the key that is closest to the found value in the sentence
                        split_input = user_input.lower().split()
                        split_input = [word.translate(str.maketrans('', '', string.punctuation)) for word in split_input]

                        # Get the positions of the value and the keys
                        val_idx = split_input.index(str(v))
                        key0_idx = split_input.index(input_keys[0])
                        key1_idx = split_input.index(input_keys[1])
                        
                        # Compute the distances
                        dist_key0 = abs(key0_idx - val_idx)
                        dist_key1 = abs(key1_idx - val_idx)
                        
                        # Determine the closest word; if both have the same distance, choose the key behind the value
                        if (dist_key0 < dist_key1) or (dist_key0 == dist_key1 and key0_idx > val_idx):
                            attr_key = input_keys[0]
                        #elif dist_key1 < dist_key0:
                        elif (dist_key0 > dist_key1) or (dist_key0 == dist_key1 and key1_idx > val_idx):
                            attr_key = input_keys[1]
                        #print(f"***input_soup(): CHOSE KEY: {attr_key}")
                    # If the current key is not in the input, skip this value (in case the other key is in the input, the value is added to it when that key is looped through)
                    elif len(input_keys) == 1 and input_keys[0] != attr_key:
                        continue
                        
                # If a value for the attribute was already found, just add the new one to it, separated by a comma
                if found_attr[attr_key] != "":
                    #print(f"***input_soup(): ATTRIBUT WURDE SCHON BELEGT MIT '{found_attr[attr_key]}'")
                    # Only add the value if it is not yet in the string
                    if not str(v) in found_attr[attr_key]:
                        found_attr[attr_key] += f", {str(v)}"
                        #print(f"***input_soup(): attribut '{attr_key}' (added): {found_attr[attr_key]}")
                else:
                    if not str(v) in found_attr[attr_key]:
                        found_attr[attr_key] += str(v)
    return found_attr

def set_filters(user_input):
    filter_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'time']
    found_attr = find_attributes(user_input, filter_attributes)
    filter_dict = {attr: val for attr, val in found_attr.items() if val != ""}
    print(f"***set_filters(): filter_dict: {filter_dict}")
    return filter_dict



###--- Recommendation Generation ---###
def input_soup(user_input):
    """
    Creates a soup from the user input that has the same scheme as the course-soups for better comparison

    Parameter:
        user_input (str): the user's input that should be transformed into a soup
    Returns:
        user_soup
    """
    print(f"\n\n***input_soup(): START CREATING SOUP")
    # Attributes that are relevant for the soup, except from title and description
    soup_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area']
    found_attr = find_attributes(user_input, soup_attributes)
    
    # Create soup  ##### MAYBE REMOVE FOUND ATTRIBUTES FROM INPUT AND ONLY PUT REST OF INPUT IN DESCRIPTION??? AT LEAST FOR ECTS, SWS & LECTURER
    soup = f"Title: . Description: {user_input}. Status: {found_attr['status']}. Mode: {found_attr['mode']}. ECTS: {found_attr['ects']}. SWS: {found_attr['sws']}. Lecturer: {found_attr['lecturer_short']}. Module: {found_attr['module']}. Area: {found_attr['area']}."
    print(f"\n***input_soup(): FINAL found_attr: {found_attr}\n***input_soup(): Soup: '{soup}'")
    return soup


def input_embedding(user_input):
    """
    Encodes a given string

        Parameters:
            user_input (str): the user's input
        Returns:
            input_emb: the embedding of the input
    """
    relevant_attributes = ['status', 'mode', 'ects', 'sws', 'lecturer_short', 'module', 'area', 'language', 'time']
    found_attr = find_attributes(user_input, relevant_attributes)

    # Get the attributes to set filters for
    filter_dict = {attr: val for attr, val in found_attr.items() if val != ""}
    print(f"***set_filters(): filter_dict: {filter_dict}")

    # Get input in soup-format
    soup = f"Title: . Description: {user_input}. Status: {found_attr['status']}. Mode: {found_attr['mode']}. ECTS: {found_attr['ects']}. SWS: {found_attr['sws']}. Lecturer: {found_attr['lecturer_short']}. Module: {found_attr['module']}. Area: {found_attr['area']}."
    print(f"\n***input_soup(): FINAL found_attr: {found_attr}\n***input_soup(): Soup: '{soup}'")

    #soup = input_soup(user_input)
    input_emb = model.encode([soup])[0]
    return input_emb, filter_dict


def recommend_courses(user_profile, rated_courses, previously_liked_courses, amount=5, filter_dict={}):
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

    #### MAYBE DELETE ALL COURSES WITH SIMILARITY BELOW THRESHOLD ALREADY HERE
    #### THEN CHECK IF LIST OF POSSIBLE RECOMMENDATIONS IS EMPTY: IF YES, SKIP REST AND WRITE RESPONSE
  
    # Delete already rated courses from top recommendations and select the specified amount of recommendations
    top_indices = [int(idx) for idx in top_courses_indices if idx not in rated_courses]

    # Get the titles of previously liked courses
    liked_titles = [past_courses[idx]['title'] for idx in previously_liked_courses]

    # Delete all titles that are already in the previously liked courses
    cleaned_indices = [idx for idx in top_indices if current_courses[idx]['title'] not in liked_titles]

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
        # Delete all courses that do not match the selected filters
        #if len(filter_dict) > 0:
            



        # Check if the similarity of any of the courses is above the threshold and select the corresponding response to return together with the list of courses to recommend
        print(f"***recommend_courses(): Best fits: {[(get_current_title(c), float(similarities[c])) for c in cleaned_indices[:amount]]}")
        for course in cleaned_indices[:amount]:
            if float(similarities[course]) >= 0.7:
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