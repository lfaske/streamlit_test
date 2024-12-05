import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from static_variables import intent_replies, instructions

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
loaded_embeddings = np.load('embeddings.npz', allow_pickle=True)
current_embeddings = loaded_embeddings['current_courses']
past_embeddings = loaded_embeddings['prev_courses']
title_embeddings = dict(loaded_embeddings['titles'].item())
intent_embeddings = dict(loaded_embeddings['intent'].item())

def get_data():
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


def get_current_title(idx):
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

def get_past_title(idx):
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


# Define the intent detection function
def detect_intent(user_input, current_state, last_recommendations):
    """
    Detects the intent of a user's input

    Parameters:
        user_input (str): the user's input
        current_state (str): the state the chatbot is currently in
        last_recommendations (list): list of the last recommended courses
    Returns:
        detected_intent: the intent it detected
        intent_replies[detected_intent] or chatbot_reply: the chatbots reply based on the intent it detected
        detected_courses: tuple (course index, x) with x being either the similarity (float) between the title and user input (if detected_intent == reference) or the feedback for the course ('liked' or 'disliked') (if detected_intent == feedback)
    """
    print("***detect_intent(): Detecting intent...")
    user_embedding = model.encode([user_input])[0]
    #max_similarity = 0
    detected_intent = "other"  # Default intent
    intent_similarities = {}
    chatbot_reply = ""
    
    # Compare user input with each intent category
    for intent, examples in intent_embeddings.items():
        # Only check those that are allowed in the current state
        #if not intent in intent_states[current_state]:
        #    intent_similarities[intent] = 0.0
        #    continue
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
    most_sim = next(iter(sorted_intents))

    # MAYBE?!? If feedback is possible in the current state, choose it if the similarity is high enough (even if it is not the most similar)
    ### -> FIRST TEST WITHOUT THIS! ## HASN'T BEEN RELEVANT YET!
    
    # If it is detected as a free description, but the probability of it being a course reference is also relatively high, try to find a matching course first
    #if most_sim[0] == 'free_description' and second_sim[0] == 'liked_course_reference' and second_sim[1] > 0.5:
    if most_sim == 'free_description' and intent_similarities['liked_course_reference'] > 0.7:
        detected_intent = "liked_course_reference"
        print("\ndetect_intent(): ***Changed intent to 'liked_course_reference'***\n")
    elif intent_similarities[most_sim] >= 0.4:
        detected_intent = most_sim
    else:
        detected_intent = "other"


    """
    for intent, examples in intent_embeddings.items():
        # Only check those that are allowed in the current state
        if not intent in intent_states[current_state]:
            continue
        similarity = cosine_similarity([user_embedding], examples).max()
        if similarity > max_similarity:
            #if detected_intent == "liked_course_reference" and intent == "free_description":

            max_similarity = similarity
            detected_intent = intent
    """

    # Threshold for valid intent detection
    #if max_similarity < 0.4:  # Adjust threshold as needed
    #    detected_intent = "other"
            
    # If the intent was detected to be a reference to a previously liked course, check which course the user is referring to
    if detected_intent == "liked_course_reference":
        print("***detect_intent(): TITLE REFERENCE DETECTED!")
        # First check if a title is (or multiple titles are) spelled out exactly ### LASSE DAS WEG; SONST WERDEN WEITERE TITEL, DIE NICHT GENAU RICHTIG GESCHRIEBEN WURDEN, IGNORIERT
        #detected_courses = []
        best_fit = ("", 0.0)
        all_titles = [c['title'] for c in past_courses]
        for title in all_titles:
            if title in user_input:
                idx = get_past_idx(title)
                print(f"***detect_intent(): Found EXACT Title: {title} (by index {idx}: {past_courses[idx]['title']})")
                #detected_courses.append((title, 1.0))  ### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
                return detected_intent, intent_replies[detected_intent], (idx, 1.0)
                
        # If any titles were directly found in the input, return them
        #if detected_courses:
        #    return intent_replies[detected_intent], detected_courses
        
        # For titles not spelled out exactly: check if parts of them are included in the input to weigh them differently   #### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
        # Check if any words from the input match words from the titles
        possible_titles = []
        ### LEMMATIZE!!! -> remove stopwords & auch Worte wie 'I', 'like', 'liked', 'course', ... (auch für similarity unten; maybe dann auch bei Titeln entfernen)
        split_input = re.sub(r'[^\w\s]', '', user_input).lower().split()
        split_titles = {t: t.lower().split() for t in all_titles}
        for title, t in split_titles.items():
            for word in split_input:
                if word in t:
                    possible_titles.append(title)
        #print("INPUT:", split_input, "TITLES:", split_titles)
        #print(f"POSSIBLE TITLES: {possible_titles}")
        
        
        # Check if the input is similar enough to any of the titles
        #print("Checking similarities of titles...")
        for title, title_emb in title_embeddings.items():  ### KANN ICH SONST AUCH ÜBER INDEX MACHEN -> BRAUCHE DANN TITLE NICHT MEHR IN TITLE_EMBEDDINGS -> IST DANN NUR NOCH LIST STATT DICT
            # If the title was already detected, skip it and continue with next title
            #if title in detected_courses:   #### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
            #    continue
            title_sim = cosine_similarity([user_embedding], [title_emb])
            #print(f"{title}: {title_sim}")
            # If a part of the title was detected in user input, set the threshold lower
            if title in possible_titles and title_sim > 0.4 and title_sim > best_fit[1]:
                #detected_courses.append((title, title_sim))   #### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
                idx = get_past_idx(title)
                print(f"***detect_intent(): Found PARTIALLY SIMILAR Title: {(title, title_sim)} -- Title by idx: {get_past_title(idx)}")
                best_fit = (idx, title_sim)
                
            elif title_sim > 0.6 and title_sim > best_fit[1]:  # Higher threshold for titles of which no part has been detected
            #if title_sim > 0.6 and title_sim > best_fit[1]:
                #detected_courses.append((title, title_sim))   #### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
                idx = get_past_idx(title)
                print(f"***detect_intent(): Found SIMILAR Title: {(title, title_sim)} -- Title by idx: {get_past_title(idx)}")
                best_fit = (idx, title_sim)
                
        # If none were found, break input down into multiple strings, separated by "and", "," and "or"
        ### IMPLEMENT ##### NO, PROB WONT DO THAT
                
        # If any similar titles were found in the input, return them
        #if detected_courses:   #### FOR NOW ONLY 1 REFERENCED COURSE ALLOWED PER MESSAGE
        #    return detected_intent, intent_replies[detected_intent], detected_courses
        if best_fit[1] > 0.5:
            return detected_intent, intent_replies[detected_intent], best_fit
        # Otherwise change the intent to free_description
        else:  ### MAYBE BESSER: VERSUCHE NÄCHST HÖCHSTEN INTENT?!? -> DAFÜR AUSFÜHRUNG DER INTENTS ALS EIGENE FUNKTIONEN (& Liste mit similarities übergeben; + default-intent, der aufgerufen wird, wenn Liste leer)?!?
            detected_intent = "free_description"

    elif detected_intent == "feedback":
        print("***detect_intent(): FEEDBACK DETECTED!")
        c_feedback = give_feedback(user_input, last_recommendations)
        if len(c_feedback) > 0:
            chatbot_reply += "You gave the following feedback:\n" #### NOT SURE OB ICH DAS DRIN LASSE ODER NUR ZUM TESTEN
            for (c, f) in c_feedback:
                chatbot_reply += f"- {current_courses[c]['title']}: {f}\n"
                #### HIER AUCH USER_PROFILE UPDATEN??? MÜSSTE DAFÜR USER_PROFILE VON CHATBOT AN DIESE FUNKTION ÜBERGEBEN & WIEDER ZURÜCKGEBEN
            chatbot_reply += "\n"
            return detected_intent, chatbot_reply, c_feedback
        else:
            #### MAYBE: IF SIMILARITY ZU FEEDBACK ÜBER BESTIMMTEN THRESHOLD: NACHRICHT AUSGEBEN, DASS ERKANNT WURDE, DASS USER FEEDBACK GEBEN WILL, ABER NICHT ERKANNT WURDE, FÜR WELCHE(N) KURS(E) BZW SENTIMENT -> KURZE BESCHREIBUNG VON FEEDBACK-FORMAT & WEITER STATUS == FEEDBACK
            ############ & FALLS UNTER THRESHOLD: NÄCHST-HÖCHSTEN INTENT VERSUCHEN
            chatbot_reply += "I think you wanted to give feedback to one or more of the recommendated courses, but I could not clearly understand you. If you want to provide feedback, please stick to the following rules: ..."
            ## RULES AUFSCHREIBEN!!!!!
            return detected_intent, chatbot_reply, []

            #detected_intent, intent_replies[detected_intent], []

              
    #if detected_intent in intent_replies.keys(): ### SHOULD NOT BE NECESSARY?!?!
    return detected_intent, intent_replies[detected_intent], []



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
        'liked' or 'disliked'
    """
    sentiment_dict = {
        "liked": ["liked", "like", "interesting", "good", "awesome", "nice", "great"],
        "disliked": ["dislike", "hate", "boring", "bad"],
        "negative": ["not", "don't", "didn't", "doesn't", "aren't", "isn't"]
    }
    # Reverse the dictionary for a more efficient lookup
    sentiment_key = {word: key for key, values in sentiment_dict.items() for word in values}

    # Split the sentence into words
    split_feedback = re.sub(r"[^\w\s']", '', sentence).lower().split()
    matches = list(set([sentiment_key[word] for word in split_feedback if word in sentiment_key]))

    # If exactly one sentiment is found (which is either 'liked' or 'disliked'), return it
    #if len(matches) == 1 and matches[0] != 'negative':
    if len(matches) == 1 and matches[0]: ### ALLOW NEGATIVE: IF SENTIMENT IN PREVIOUS SENTENCE: USE OPPOSITE
        return matches[0]
    # If only 'negative' and 'liked' are found (e.g., "I don't like course 1"), return 'disliked'
    # Not the other way around, because phrases like "I don't hate it" don't necessarily mean that it's liked
    elif len(matches) == 2 and 'negative' in matches and 'liked' in matches:
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
        sentiment = sentence_sentiment
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
        c_sentiment = sentence_sentiment(s)
        # If no course was found, skip the sentence
        if len(courses) == 0:  
            continue

        # If the sentiment is 'negative', use the opposite of the previous sentence's sentiment (if any)
        if c_sentiment == 'negative' and len(given_feedback) > 0:
            print(f"***give_feedback(): Found feedback {c_sentiment} in sentence {s}!")
            c_sentiment = given_feedback[-1][1]
            print(f"*** -> Sentiment of last sentence: {given_feedback[-1][1]}")
            c_sentiment = 'liked' if given_feedback[-1][1] == 'disliked' else 'disliked'
            print(f"*** -> Sentiment of current sentence changed to {c_sentiment}")
        # If it is negative but there have not yet been sentences with a detected sentiment, skip the sentence
        elif c_sentiment == 'negative':  
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
def update_user_profile(user_profile, pref_embedding = None, rated_course = None, liked=True, learning_rate=0.1):
    """
    Updates the user's preferences according to a free description, a reference to a liked course, or given feedback

    Parameters: 
        user_profile (array?): current embedding of the user's preferences
        pref_embedding (array?)*: embedding of either the user's description, a previously liked course, or a course the user gave feedback for
        rated_course ((int, str))*: either (index, 'past') of a previously liked course or (index, 'current') of a course the user gave feedback for
        liked (bool): if the user liked the course / the recommendation of the course; for descriptions always True
        learning_rate (float): how strong the feedback should influence the user's preferences
        * either pref_embedding or rated_course is necessary
    Returns: 
        updated user_profile
    """
    print("***update_user_profile(): Updating user profile...")
    if pref_embedding is None:
        print(f"***update_user_profile(): Using: {rated_course}...")
        if rated_course[1] == 'past':
            #course_idx = [c['title'] for c in past_courses].index(rated_course[0]) ### ONLY NECESSARY, IF COURSES SAVED WITH TITLE INSTEAD OF IDX
            #pref_embedding = past_embeddings[course_idx]
            pref_embedding = past_embeddings[rated_course[0]]  ## IF COURSES SAVED BY IDX
        else:
            #course_idx = [c['title'] for c in current_courses].index(rated_course[0]) ### ONLY NECESSARY, IF COURSES SAVED WITH TITLE INSTEAD OF IDX
            #pref_embedding = current_embeddings[course_idx]
            pref_embedding = current_embeddings[rated_course[0]]  ## IF COURSES SAVED BY IDX
    if user_profile is not None:
        if liked:  # Positive feedback, liked course or free description
            user_profile += learning_rate * pref_embedding  # Make the embedding of the user's preferences more similar to the liked course
        else:      # Negative feedback
            user_profile -= learning_rate * pref_embedding  # Make the embedding of the user's preferences less similar to the disliked course
    else: ### CREATE A CASE FOR NO PROFILE & DISLIKED??
        # If no profile exists, set user's embedding to the liked course's embedding (this is the case, if a user starts by refering to a previously liked course)
        user_profile = pref_embedding
    ### DO I HAVE TO RETURN IT?? OR CAN I MANAGE (SAVE & CHANGE) IT DIRECTLY IN THIS FILE?? -> I think I can't?!?
    return user_profile


###--- Recommendation Generation ---###

def input_embedding(user_input):
    """
    Encodes a given string

        Parameters:
            user_input (str): the user's input
        Returns:
            input_emb: the embedding of the input
    """
    input_emb = model.encode([user_input])[0]
    return input_emb


def recommend_courses(user_profile, rated_courses, previously_liked_courses, amount=5, filter=None):
    """
    Generates recommendations based on a user's preferences.

        Parameters: 
            user_profile (array?): embedding of the user's preferences
            rated_courses (list): indices of courses (in current_courses) the user has already rated
            previously_liked_courses (list): courses the user has stated to have liked (from past semesters)
            amount (int): how many courses should be recommended - default: top 5
            filter (dict): all selected filters (IF IMPLEMENTED)
        Returns: 
            list of tuples (recommended course index, similarity to user profile)
    """
    # Compute cosine similarity between the user profile and course embeddings
    similarities = cosine_similarity([user_profile], current_embeddings)[0]

    # Rank courses by similarity
    #top_courses_indices = similarities.argsort()[-amount:][::-1]  ## AMOUNT AFTER DELETING ALREADY RATED INDICES
    top_courses_indices = similarities.argsort()[::-1]

    liked_titles = [past_courses[idx]['title'] for idx in previously_liked_courses]

    # Delete already rated courses from top recommendations and select the specified amount of recommendations
    #top_indices = [int(idx) for idx in top_courses_indices if idx not in rated_courses][:amount]
    top_indices = [int(idx) for idx in top_courses_indices if idx not in rated_courses]
    print(f"***recommend_courses(): Top recommendations that were liked in the past: {[idx for idx in top_indices if past_courses[idx]['title'] in liked_titles]}")
    cleaned_indices = [idx for idx in top_indices if past_courses[idx]['title'] not in liked_titles][:amount]

    top_indices_sim = [(idx, float(similarities[idx])) for idx in cleaned_indices]

    #top_courses_indices_sim = [float(similarities[idx]) for idx in top_courses_indices]  ### JUST FOR TESTING
    #top_courses = [current_courses[i] for i in top_courses_indices]   ### USE IDX INSTEAD; JUST FOR TESTING
    #print(f"\ntop_courses_indices: {top_courses_indices}\ntop_courses_indices_sim: {top_courses_indices_sim}\ntop_courses: {[c['title'] for c in top_courses]}")
    #print(f"WITH NEW TECHNIQUE: {top_indices_sim}")
    
    #return list(zip(top_courses, top_courses_indices_sim))  ### NOT NECESSARY IF USING JUST IDX
    return top_indices_sim


def write_recommendation(recommended_courses):
    """
    Decides whether or not to recommend generated courses based on the similarity

    Parameters:
        recommended_courses (list of tuples): top 5 recommendations + their similarities to the user preferences
    Returns:
        chatbot message
        to_recommend: indices of the recommended courses
    """
    response = ""
    to_recommend = []
    #print(f"***Thinking about recommending one of these: {[get_current_title(c[0]) for c in recommended_courses]}")
    #new_rec_courses = [c for c in recommended_courses]
    for (c, sim) in recommended_courses:
        if sim >= 0.333:
            to_recommend.append(c)
            #print(f"***Good match: {get_current_title(c)} -- {sim}")
    #print(f"\n***TO RECOMMEND: {to_recommend} (LEN: {len(to_recommend)})***\n")
    if len(to_recommend) > 1:
        response = "I found some courses you might like:  \n"
        #rec_string = "\n".join([get_current_title(c[0]) for c in recommended_courses])
        rec_string = ""
        for idx, c in enumerate(recommended_courses, start = 1):
            rec_string += f"{idx}: {get_current_title(c[0])}  \n"
        response += rec_string
        response += f"\nPlease tell me if these courses sound interesting to you.  \nHint: {instructions['feedback']} "
    elif len(to_recommend) == 1:
        response = f"I found a course you might like:  \n- {get_current_title(to_recommend[0])}\nFeedback would help a lot to improve further recommendations. Please tell me if this course sounds interesting or not. "
    else:
        response = "I need some more information to generate good recommendations for you. Could you tell me more about what kind of course you are looking for? Or is there any course you liked in the past that you didn't tell me about yet? "
    
    return response, to_recommend