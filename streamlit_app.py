import streamlit as st
from responses import response_generator
from recommender import detect_intent, update_user_profile, recommend_courses, input_embedding, get_current_title, get_past_title, write_recommendation
import re
from static_variables import confirmation_dict, confirmation_replies, abbreviations, hints

st.title("Test Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the current state
if "current_state" not in st.session_state:
    st.session_state.current_state = 'default'

# Initialize the user profile
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        'preferences': None,
        'previously_liked_courses': [],
        'rated_courses': [],
        'last_recommendations': []
    }

# Initialize the instruction/hint buttons
if 'hint_button' not in st.session_state:
    st.session_state.hint_button = {
        'show_instructions': False,
        'show_hint': False,
        'current_hint': 'none'
    }

# Function to toggle the instruction/hint buttons
def toggle_hint(hint_key):
    # Close the other hint before opening the clicked one
    for key in ['show_instructions', 'show_hint']:
        if key != hint_key:
            st.session_state.hint_button[key] = False
    # Toggle the selected hint
    st.session_state.hint_button[hint_key] = not st.session_state.hint_button[hint_key]

#### DO I NEED THAT??? TEST WITHOUT
# Initialize course data and embeddings (including intent embeddings)
#if "data" not in st.session_state:
#    st.session_state.data = get_data()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        #st.session_state.messages[-1]['state']

#with st.chat_message("supervisor"): # parameters "user" or "assistant" (for preset styling & avatar); otherwise can name it how I want
# FOR TESTING
#if len(st.session_state.messages) > 0:
   # st.write(f"{st.session_state.messages[-1]['content']}, {st.session_state.messages[-1]['state']}")

# Accept user input
if user_input := st.chat_input("Start typing ..."):
    if user_input == "reset":
        st.session_state.messages = []
        st.session_state.current_state = 'default'
        st.session_state.user_profile = {
            'preferences': None,
            'previously_liked_courses': [],
            'rated_courses': [],
            'last_recommendations': []
            }
        st.session_state.hint_button = {
            'show_instructions': False,
            'show_hint': False,
            'current_hint': 'none'
        }
    if user_input == "start":
        st.session_state.hint_button['current_hint'] = 'How to start'
    else:
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Expand abbreviations in input
        for abbrev, full_form in abbreviations.items():
            user_input = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, user_input, flags=re.IGNORECASE)

        chatbot_reply = ""
               

        # Check state
        # The chatbot asked the user if it detected the correct course referenced by the user and waits for confirmation
        if st.session_state.current_state == 'confirmation':
            # Check if the user confirmed or denied
            confirmation_keys = []
            for key, value_list in confirmation_dict.items():
                if any(value in user_input for value in value_list):
                    confirmation_keys.append(key)

            #if user_input.lower() in confirmation_dict['yes']:
            if len(confirmation_keys) == 1 and confirmation_keys[0] == 'yes':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['yes']
                liked_course = st.session_state.user_profile['previously_liked_courses'][-1]
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], rated_course=(liked_course, 'past'), liked=True)

                # Generate recommendations
                new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])
                reply, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                chatbot_reply += reply
                
            #elif user_input.lower() in confirmation_dict['no']:
            elif len(confirmation_keys) == 1 and confirmation_keys[0] == 'no':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['no']
                del st.session_state.user_profile['previously_liked_courses'][-1]
            else:
                chatbot_reply = confirmation_replies['other']
        
        
        # default or feedback: User can describe courses or reference a liked course; if state == feedback, they can also rate the previous recommendations
        else: 
            # Detect intent
            #f"Type of st.session_state.user_profile: {type(st.session_state.user_profile)}"
            detected_intent, correct_reply, detected_courses = detect_intent(user_input, st.session_state.current_state, st.session_state.user_profile['last_recommendations'])
            chatbot_reply += correct_reply

            # If the user referred to a liked course, update their profile and generate new recommendations 
            if detected_intent == "liked_course_reference":
                #### ONLY ALLOW 1 REFERENCED COURSE PER MESSAGE FOR NOW!! ANSATZ FÜR MEHRERE: SIEHE BACKUP
                st.session_state.user_profile['previously_liked_courses'].append(detected_courses[0])
                # If the certainty (similarity of the title to the user input) is not high enough: ask if correct
                if detected_courses[1] < 0.75:
                    chatbot_reply = f"I'm not sure if I understood you correctly. You liked the course {get_past_title(detected_courses[0])}, is that correct? "
                    st.session_state.current_state = "confirmation"
                else:
                    chatbot_reply = f"You liked the course {get_past_title(detected_courses[0])}. "
                    ### TRY RECOMMENDING

            # If the user gave a description, update their profile and generate new recommendations 
            elif detected_intent == "free_description":
                #print("*** GAVE FREE DESCRIPTION!")
                input_emb = input_embedding(user_input)  # Compute the embedding of the user's input
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], pref_embedding=input_emb, liked=True)
                new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])  # Generate new recommendations
                #chatbot_reply += write_recommendation(new_recommendations)
                reply, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                chatbot_reply += reply

            # If the user gave feedback, find out for which recommended course and update the user profile accordingly
            elif detected_intent == "feedback":
                ## IF NOT ALREADY DONE IN RECOMMENDER.PY:
                #rated_recommendations = give_feedback(user_input, last_recommendations)
                #if len(rated_recommendations) == 0:
                #    chatbot_reply += f"I didn't understand you correctly. I thought you wanted to give feedback, but I couldn't detect it correctly. If you want to give feedback, refer to the courses by their position in the list of recommendations (e.g., 'course 1', 'first', 'second'). If you want to give both positive and negative feedback, please write it into different sentences or separate the sentences with 'but'. "
                #else:
                #    for c, sentiment in rated_recommendations.items():
                #        user_profile = update_user_profile(user_profile, rated_course = (c, 'current'), liked = sentiment == 'liked', learning_rate=0.1)
                #        rated_courses.append(c)
                #        print(f"***Updated profile with: {c} -> {sentiment}")

                ## IF DONE IN RECOMMENDER.PY:
                if len(detected_courses) > 0:
                    print(f"xxx INTENT FEEDBACK: LEN(DETECTED_COURSES) == {len(detected_courses)}")
                    for (c, sentiment) in detected_courses:
                        st.session_state.user_profile['preferences'] = update_user_profile(st.session_state.user_profile['preferences'], rated_course = (c, 'current'), liked = (sentiment == 'liked'))
                        st.session_state.user_profile['rated_courses'].append(c)
                        print(f"xxx intent feedback: Updated profile with: {get_current_title(c)} -> {sentiment}")
                    # Generate new recommendations
                    new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])
                    #chatbot_reply += write_recommendation(new_recommendations)
                    reply, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                    chatbot_reply += reply
                
                #detected_intent, chatbot_reply, [c_feedback] --- detected_intent, correct_reply, detected_courses

        

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.markdown(chatbot_reply)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": chatbot_reply})

        # Adding a 'Hint' field with a hover tooltip
        #hint_text = "Hint"
        #tooltip = "This is a helpful hint to assist you."

        # Creating a hoverable 'Hint' field
        #st.markdown(f"""
        #<div style="border: 1px solid #007bff; padding: 5px 10px; font-size: 10px; width: fit-content; border-radius: 5px; display: inline-block;">
        #    <span title="{tooltip}" style="color: grey; text-decoration: underline;">{hint_text}</span>
        #</div>
        #""", unsafe_allow_html=True)

# Adding a 'Hint' button to toggle the expanded text
#st.markdown("""
#    <style>
#        .small-button button {
#            font-size: 10px;
#            padding: 5px 10px;
#            border-radius: 5px;
#        }
#    </style>
#""", unsafe_allow_html=True)

#st.markdown("""
#    <style>
#        .button-container {
#            display: flex;
#            justify-content: start; /* Align buttons to the left */
#            gap: 5px; /* Adjust the spacing between buttons */
#        }
#        .button-container .stButton button {
#            font-size: 12px;
#            padding: 5px 10px;
#            border-radius: 5px;
#        }
#    </style>
#""", unsafe_allow_html=True)

# Custom CSS for the box
st.markdown("""
    <style>
        .hint-box {
            border: 1px solid grey;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
            font-size: 10px;
            max-width: 80%;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

#f"Current hint: {st.session_state.hint_button['current_hint']}"
if st.session_state.hint_button['current_hint'] == 'none':
    # Adding the 'Hint' button to toggle the expanded text
    hint_button = st.button('Instructions')

    # Toggle the visibility of the hint text
    if hint_button:
        st.session_state.hint_button['show_hint'] = not st.session_state.hint_button['show_hint']

    # Check if the button is clicked and display the hint
    if st.session_state.hint_button['show_hint']:
        st.markdown(f"""
            <div class="hint-box">
                {hints['instructions']}
            </div>
        """, unsafe_allow_html=True)
else:

    # Layout for buttons side by side
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("Instructions", key="button_1"):
            toggle_hint("show_instructions")

    with col2:
        if st.button(st.session_state.hint_button['current_hint'], key="button_2"):
            toggle_hint("show_hint")

    # Display the corresponding expanded text
    if st.session_state.hint_button['show_instructions']:
        st.markdown(f"""
            <div class="hint-box">
                {hints['instructions']}
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.hint_button['show_hint']:
        st.markdown(f"""
            <div class="hint-box">
                {hints[st.session_state.hint_button['current_hint']]}
            </div>
        """, unsafe_allow_html=True)


