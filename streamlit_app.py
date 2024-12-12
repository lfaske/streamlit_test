import streamlit as st
from responses import response_generator
from recommender import detect_intent, update_user_profile, recommend_courses, input_embedding, get_current_title, get_past_title, write_recommendation, get_five_courses, get_details
import re
from static_variables import confirmation_dict, confirmation_replies, abbreviations, hints

st.title("Test Chat")

# Initialize chat history
welcome_msg = """
Hey, nice to meet you!  \nI'm here for helping you find interesting courses for your next semester. Just name a course you liked in the past or describe what kind of course you're looking for. For more instructions, click on the button below."""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg, "content_end": "", "recommended_courses": []}]

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
        'current_hint': 'all'
    }

if "show_details" not in st.session_state:
    st.session_state.show_details = {}

# Function to toggle the instruction/hint buttons
def toggle_hint(hint_key):
    # Close the other hint before opening the clicked one
    for key in ['show_instructions', 'show_hint']:
        if key != hint_key:
            st.session_state.hint_button[key] = False
    # Toggle the selected hint
    st.session_state.hint_button[hint_key] = not st.session_state.hint_button[hint_key]

def display_recommendations(courses, response, response_end):
    print(f"Displaying recommendations: {[c['title'] for c in courses]}")
    response += "Here are some course recommendations: "
    course_buttons = [{"title": course["title"], "details": course} for course in courses]
    
    # Append to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "content_end": response_end, "buttons": course_buttons})

# Function to render a chat message
def render_message(msg, msg_idx):
    if msg["role"] == "assistant":
        # Display chatbot's text
        st.markdown(msg["content"])
        
        # Check if there are buttons to render
        if "buttons" in msg:
            for button_data in msg["buttons"]:
                title = button_data["title"]
                details = button_data["details"]

                button_key = f"button_{msg_idx}_{title}"
                
                # Initialize visibility for each button
                if button_key not in st.session_state.show_details:
                    st.session_state.show_details[button_key] = False
                
                # Create button
                toggle = st.button(f"➕ {title}", key=button_key)
                
                # Toggle visibility
                if toggle:
                    st.session_state.show_details[button_key] = not st.session_state.show_details[button_key]
                
                # Display details if visible
                if st.session_state.show_details[button_key]:
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                            <strong>Description:</strong> {details['description']}<br>
                            <strong>ECTS:</strong> {details['ects']}<br>
                            <strong>Lecturer:</strong> {', '.join(details['lecturer'])}
                        </div>
                    """, unsafe_allow_html=True)
        # Display assistant's text
        if msg["content_end"] != "":
            st.markdown(msg["content_end"])
    else:
        st.markdown(msg["content"])

def chatbot_response(response, response_end, courses=[]):
    with st.chat_message("assistant"):
        # Display chatbot reply
        st.markdown(response)

        # Display each course recommendation as a list with buttons
        for course in courses:
            title = course["title"]

            # Create a horizontal layout with a button and title
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                toggle_button = st.button("➕", key=f"button_{title}")
            with col2:
                st.markdown(f"**{title}**")

            # Toggle visibility on button click
            if toggle_button:
                st.session_state.show_details[title] = not st.session_state.show_details[title]

            # Display course details if visible
            if st.session_state.show_details[title]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <strong>Description:</strong> {course['description']}<br>
                    <strong>ECTS:</strong> {course['ects']}<br>
                    <strong>Lecturer:</strong> {course['lecturer']}
                </div>
                """, unsafe_allow_html=True)
                


# Display chat messages from history on app rerun
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg['role']):
        render_message(msg, i)

#with st.chat_message("supervisor"): # parameters "user" or "assistant" (for preset styling & avatar); otherwise can name it how I want
# FOR TESTING
#if len(st.session_state.messages) > 0:
   # st.write(f"{st.session_state.messages[-1]['content']}, {st.session_state.messages[-1]['state']}")

# FOR TESTING Simulate a user query and bot response
#if st.button("Simulate Recommendation"):
#    courses = get_five_courses()
#    display_recommendations(courses, "This is a test", "")

# Accept user input
if user_input := st.chat_input("Start typing ..."):
    print(f"\n\n---------------------------------------------------------\n\n")
    ### v JUST FOR TESTING v ###
    print(st.session_state.messages)
    if user_input == "reset":
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg, "content_end": "", "buttons": []}]
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
            'current_hint': 'all'
        }
        st.session_state.show_details = {}
    elif user_input == "start":
        st.session_state.hint_button['current_hint'] = 'How to start'
    ### ^ JUST FOR TESTING ^ ###

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
        chatbot_reply_end = ""
        recommended_courses = []
               

        # Check state
        # The chatbot asked the user if it detected the correct course referenced by the user and waits for confirmation
        if st.session_state.current_state == 'confirmation':
            # Check if the user confirmed or denied
            confirmation_keys = []
            for key, value_list in confirmation_dict.items():
                if any(value in user_input.lower() for value in value_list):
                    confirmation_keys.append(key)

            #if user_input.lower() in confirmation_dict['yes']:
            if len(confirmation_keys) == 1 and confirmation_keys[0] == 'yes':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['yes']
                liked_course = st.session_state.user_profile['previously_liked_courses'][-1]
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], rated_course=(liked_course, 'past'), liked=True)

                # Generate recommendations
                new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])
                reply, reply_end, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                chatbot_reply += reply
                if len(st.session_state.user_profile['last_recommendations']) > 0:
                    recommended_courses = st.session_state.user_profile['last_recommendations']
                ##### IF THERE ARE ANY RECOMMENDATIONS: 
                ########## ADD THEM TO REPLY (INCLUDING FIELD TO OPEN DETAILS)
                ########## ADD reply_end TO REPLY
                
            #elif user_input.lower() in confirmation_dict['no']:
            elif len(confirmation_keys) == 1 and confirmation_keys[0] == 'no':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['no']
                del st.session_state.user_profile['previously_liked_courses'][-1]
            else:
                chatbot_reply = confirmation_replies['other']
        
        
        # default: User can describe courses, reference a liked course, or rate the previous recommendations
        else: 
            # Detect intent
            #f"Type of st.session_state.user_profile: {type(st.session_state.user_profile)}"
            detected_intent, correct_reply, detected_courses = detect_intent(user_input, st.session_state.user_profile['last_recommendations'])
            chatbot_reply += correct_reply

            # If the user referred to a liked course, update their profile and generate new recommendations 
            if detected_intent == "liked_course_reference":
                #### ONLY ALLOW 1 REFERENCED COURSE PER MESSAGE FOR NOW!! ANSATZ FÜR MEHRERE: SIEHE BACKUP
                st.session_state.user_profile['previously_liked_courses'].append(detected_courses[0])
                # If the certainty (similarity of the title to the user input) is not high enough: ask if correct
                if detected_courses[1] < 0.7:
                    chatbot_reply = f"I'm not sure if I understood you correctly. You liked the course {get_past_title(detected_courses[0])}, is that correct? "
                    st.session_state.current_state = "confirmation"
                else:
                    chatbot_reply = f"You liked the course {get_past_title(detected_courses[0])}.  \n"
                    st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], rated_course=(detected_courses[0], 'past'), liked=True)
                    
                    # Generate recommendations
                    new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])
                    reply, reply_end, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                    chatbot_reply += reply
                    if len(st.session_state.user_profile['last_recommendations']) > 0:
                        recommended_courses = st.session_state.user_profile['last_recommendations']
                    ##### IF THERE ARE ANY RECOMMENDATIONS: 
                    ########## ADD THEM TO REPLY (INCLUDING FIELD TO OPEN DETAILS)
                    ########## ADD reply_end TO REPLY

            # If the user gave a description, update their profile and generate new recommendations 
            elif detected_intent == "free_description":
                #print("*** GAVE FREE DESCRIPTION!")
                input_emb = input_embedding(user_input)  # Compute the embedding of the user's input
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], pref_embedding=input_emb, liked=True)
                new_recommendations = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'])  # Generate new recommendations
                #chatbot_reply += write_recommendation(new_recommendations)
                reply, reply_end, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                chatbot_reply += reply
                if len(st.session_state.user_profile['last_recommendations']) > 0:
                    recommended_courses = st.session_state.user_profile['last_recommendations']
                ##### IF THERE ARE ANY RECOMMENDATIONS: 
                ########## ADD THEM TO REPLY (INCLUDING FIELD TO OPEN DETAILS)
                ########## ADD reply_end TO REPLY

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
                    reply, reply_end, st.session_state.user_profile['last_recommendations'] = write_recommendation(new_recommendations)
                    chatbot_reply += reply
                    if len(st.session_state.user_profile['last_recommendations']) > 0:
                        recommended_courses = st.session_state.user_profile['last_recommendations']
                    ##### IF THERE ARE ANY RECOMMENDATIONS: 
                    ########## ADD THEM TO REPLY (INCLUDING FIELD TO OPEN DETAILS)
                    ########## ADD reply_end TO REPLY
                
                #detected_intent, chatbot_reply, [c_feedback] --- detected_intent, correct_reply, detected_courses

        

        # Display assistant response in chat message container
        #with st.chat_message("assistant"):
        #    response = st.markdown(chatbot_reply)

        #chatbot_response(chatbot_reply, chatbot_reply_end, courses=[])

        # Add assistant response to chat history
        #st.session_state.messages.append({"role": "assistant", "content": chatbot_reply})

        course_buttons = [{"title": course["title"], "details": course} for course in [get_details(c) for c in recommended_courses]]
        st.session_state.messages.append({"role": "assistant", "content": chatbot_reply, "content_end": chatbot_reply_end, "buttons": course_buttons})
        print(f"--- CHATBOT_MSG: {st.session_state.messages[-1]}")
        msg_idx = len(st.session_state.messages[-1])-1
        with st.chat_message("assistant"):
            render_message(st.session_state.messages[-1], msg_idx)






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

elif st.session_state.hint_button['current_hint'] == 'all':
    # Layout for buttons side by side
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Instructions", key="button_1"):
            toggle_hint("show_instructions")

    with col2:
        if st.button("Feedback Hint", key="button_2"):
            toggle_hint("show_hint")
            
    #with col3:
    #    if st.button(st.session_state.hint_button['current_hint'], key="button_2"):
    #        toggle_hint("show_hint")

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
                {hints['Feedback Hint']}
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



