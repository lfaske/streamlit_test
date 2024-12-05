## Contains all functions that deal with input processing and reply generation
from recommender import detect_intent, update_user_profile, recommend_courses, input_embedding, get_current_title, get_past_title
from static_variables import confirmation_dict, confirmation_replies, abbreviations, instructions
import re

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
        response = "I found some courses you might like:\n"
        #rec_string = "\n".join([get_current_title(c[0]) for c in recommended_courses])
        rec_string = ""
        for idx, c in enumerate(recommended_courses, start = 1):
            rec_string += f"{idx}: {get_current_title(c[0])}\n"
        response += rec_string
        response += f"\nPlease tell me if these courses sound interesting to you.\nHint: {instructions['feedback']} "
    elif len(to_recommend) == 1:
        response = f"I found a course you might like:\n- {get_current_title(to_recommend[0])}\nFeedback would help a lot to improve further recommendations. Please tell me if this course sounds interesting or not. "
    else:
        response = "I need some more information to generate good recommendations for you. Could you tell me more about what kind of course you are looking for? Or is there any course you liked in the past that you didn't tell me about yet? "
    
    return response, to_recommend


