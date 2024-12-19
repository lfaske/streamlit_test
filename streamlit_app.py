import streamlit as st
from recommender import detect_intent, update_user_profile, recommend_courses, input_embedding, get_current_title, get_past_title, get_details, get_all_filter
import re
from chatbot_variables import confirmation_dict, confirmation_replies, abbreviations, hints, help_text

st.title("Course Recommender Bot")


###--- Initialize session_state variables ---###

# Initialize the chat history with a welcome message from the chatbot
welcome_msg = "Hey, nice to meet you!  \nI'm CouRe, a chatbot designed to help you find interesting courses for the upcoming semester. You can start by either naming a course you liked in the past or describe what kind of course you are looking for. For more detailed instructions, click on the button below."
#here for helping you find interesting courses for your next semester. Just name a course you liked in the past or describe what kind of course you're looking for. For more instructions, click on the button below."
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg, "content_end": "", "new_recommendations": []}]

# Initialize the current state
if "current_state" not in st.session_state:
    st.session_state.current_state = 'default'

# Initialize the user profile
if "user_profile" not in st.session_state:
    all_filter_attr = get_all_filter()
    st.session_state.user_profile = {
        'preferences': None,
        'previously_liked_courses': [],
        'rated_courses': [],
        'last_recommendations': []
    }

# Initialize the dictionary for filters
if "filter_dict" not in st.session_state:
    all_filters = get_all_filter()
    st.session_state.filter_dict = {key: {val: [] for val in all_filters[key]} if key == 'filter_time' else [] for key in all_filters}

# Initialize the status for the times filter (default: disabled)
#if "disable_times" not in st.session_state:
#    all_days = list(st.session_state.filter_dict['filter_time'].keys())
#    st.session_state.disable_times = {}
#    for day in all_days:
#        day_key = f"disable_times_{day}"
#        st.session_state[day_key] = False

if "allow_days" not in st.session_state:
    all_days = list(st.session_state.filter_dict['filter_time'].keys())
    st.session_state.allow_days = {}
    for day in all_days:
        day_key = f"allow_day_{day}"
        st.session_state[day_key] = False

# Initialize the instruction/hint buttons
if 'hint_button' not in st.session_state:
    st.session_state.hint_button = {
        'show_instructions': False,
        'show_hint': False,
        'current_hint': 'all'  #### PROB UNNÖTIG --- BRAUCHE ICH NUR, WENN ICH VISIBILITY VON HINT-BUTTONS VERÄNDERN MÖCHTE (sodass immer nur aktuell relevante Buttons angezeigt werden); ist wahrscheinlich größtenteils überflüssig & daher zu viel Aufwand
    }

# Initialize the course details
if "show_details" not in st.session_state:
    st.session_state.show_details = {}

# Initialize the sidebar  ### PROB UNNÖTIG
if "sidebar" not in st.session_state:
    all_filter_attr = get_all_filter()
    st.session_state.sidebar = {key: [] for key in all_filter_attr}




#st.selectbox('test me',(1,2,3),index=1,key='sel_key') # no explicit session_state here!




###--- Handle buttons and messages ---###

# Allows to toggle the instruction/hint buttons
def toggle_hint(hint_key):
    # Close the other hint before opening the clicked one
    for key in ['show_instructions', 'show_hint']:
        if key != hint_key:
            st.session_state.hint_button[key] = False
    # Toggle the selected hint
    st.session_state.hint_button[hint_key] = not st.session_state.hint_button[hint_key]


# Render a chat message
def render_message(msg, msg_idx):
    if msg["role"] == "assistant":
        # Display the chatbot's message
        st.markdown(msg["content"])
        
        # Check if there are buttons (to open details of recommended courses) to render
        if "buttons" in msg:
            for button_nr, button_data in enumerate(msg["buttons"], 1):
                title = button_data["title"]
                details = button_data["details"]

                button_key = f"button_{msg_idx}_{title}"
                
                # Initialize visibility for each button
                if button_key not in st.session_state.show_details:
                    st.session_state.show_details[button_key] = False
                
                # Create the button
                #toggle = st.button(f"➕ {title}", key=button_key)
                toggle = st.button(f"{button_nr}. {title}", key=button_key)
                
                # Toggle visibility
                if toggle:
                    st.session_state.show_details[button_key] = not st.session_state.show_details[button_key]
                
                # Display details if visible
                if st.session_state.show_details[button_key]:
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                            <strong>Course type:</strong> {details['status']}<br>
                            <strong>Mode:</strong> {details['mode']}<br>
                            <strong>Language:</strong> {details['language']}<br>
                            <strong>ECTS:</strong> {details['ects']}<br>
                            <div style="margin-bottom: 10px;"><strong>SWS:</strong> {details['sws']}</div>
                            <div style="margin-bottom: 10px;"><strong>Prerequisites:</strong> <br>{details['prerequisites']}</div>
                            <div style="margin-bottom: 10px;"><strong>Lecturer:</strong> <br>{'<br>'.join(details['lecturer'])}</div>
                            <div style="margin-bottom: 10px;"><strong>Times:</strong> <br>{details['time']}</div>
                            <div style="margin-bottom: 10px;"><strong>Module assignment:</strong> <br>{'<br>'.join(details['module'])}</div>
                            <div><strong>Description:</strong> <br>{details['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        # Display assistant's text
        if msg["content_end"] != "":
            st.markdown(msg["content_end"])
    else:
        st.markdown(msg["content"])


# Display chat messages from history on app rerun
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg['role']):
        render_message(msg, i)

# Get user input
if user_input := st.chat_input("Start typing ..."):

    ### v JUST FOR TESTING v ###
    print(f"\n\n---------------------------------------------------------\n\n")
    #print(st.session_state.messages)
    if user_input == "reset":
        all_filter_attr = get_all_filter()
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
        st.session_state.sidebar = {key: [] for key in all_filter_attr}
        all_filters = get_all_filter()
        st.session_state.filter_dict = {key: {val: [] for val in all_filters[key]} if key == 'filter_time' else [] for key in all_filters}
    elif user_input == "start":
        st.session_state.hint_button['current_hint'] = 'How to start'
    ### ^ JUST FOR TESTING ^ ###

    else:  ## <- UNNÖTIG, WENN TESTING-PART WEG IST
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add the message to the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Expand abbreviations in input
        for abbrev, full_form in abbreviations.items():
            user_input = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, user_input, flags=re.IGNORECASE)

        chatbot_reply = ""
        chatbot_reply_end = ""  # E.g. content after presenting recommended courses
        new_recommendations = []
               

        # If the chatbot asked the user whether or not the detected course reference was correct, it waits for confirmation
        if st.session_state.current_state == 'confirmation':

            # Look for words marking confirmation or negation
            confirmation_keys = []
            for key, value_list in confirmation_dict.items():
                if any(value in user_input.lower() for value in value_list):
                    confirmation_keys.append(key)

            # If a confirmation is found, set the current state back to default, update the user profile and generate recommendations
            if len(confirmation_keys) == 1 and confirmation_keys[0] == 'yes':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['yes']
                liked_course = st.session_state.user_profile['previously_liked_courses'][-1]

                # Update the user profile and generate recommendations
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], rated_course=(liked_course, 'past'), liked=True)
                reply, reply_end, st.session_state.user_profile['last_recommendations'] = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'], filter_dict=st.session_state.filter_dict)
                chatbot_reply += reply
                chatbot_reply_end += reply_end
                new_recommendations = st.session_state.user_profile['last_recommendations']
                
            # If a negation is found, set the current state back to default and remove the last entry from the previously liked courses
            elif len(confirmation_keys) == 1 and confirmation_keys[0] == 'no':
                st.session_state.current_state = 'default'
                chatbot_reply = confirmation_replies['no']
                del st.session_state.user_profile['previously_liked_courses'][-1]

            # If neither is found, ask the user to first confirm or deny, before giving new information 
            else:
                chatbot_reply = confirmation_replies['other']
        
        # If the current state is default
        else: 
            # Detect the intent of the message
            detected_intent, correct_reply, detected_courses = detect_intent(user_input, st.session_state.user_profile['last_recommendations'])
            chatbot_reply += correct_reply

            # If the user referred to a liked course, add the course to the previously liked courses 
            if detected_intent == "liked_course_reference":
                detected_courses = detected_courses[0]
                st.session_state.user_profile['previously_liked_courses'].append(detected_courses[0])

                # If the certainty (similarity of the title to the user input) is not high enough: ask if correct
                if detected_courses[1] < 0.7:
                    chatbot_reply = f"I'm not sure if I understood you correctly. You liked the course {get_past_title(detected_courses[0])}, is that correct? "
                    st.session_state.current_state = "confirmation"

                # Otherwise, update the user profile and generate recommendations
                else:
                    chatbot_reply = f"You liked the course {get_past_title(detected_courses[0])}.  \n"
                    st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], rated_course=(detected_courses[0], 'past'), liked=True)
                    reply, reply_end, st.session_state.user_profile['last_recommendations'] = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'], filter_dict=st.session_state.filter_dict)
                    chatbot_reply += reply
                    chatbot_reply_end += reply_end
                    new_recommendations = st.session_state.user_profile['last_recommendations']

            # If the user gave a description, update their profile and generate new recommendations 
            elif detected_intent == "free_description":
                input_emb, st.session_state.filter_dict = input_embedding(user_input, st.session_state.filter_dict)  # Compute the embedding of the user's input and get the mentioned attributes for the filter
                #input_emb, filter_updates = input_embedding(user_input, st.session_state.filter_dict)  # Compute the embedding of the user's input and get the mentioned attributes for the filter
                
                # Check if the user set a time for a new day in the input
                for day, d_time in st.session_state.filter_dict['filter_time'].items():
                    if d_time != [] and not st.session_state[f"allow_day_{day}"]:
                        st.session_state[f"allow_day_{day}"] = True


                # Update the user profile and generate recommendations
                st.session_state.user_profile['preferences'] = update_user_profile(user_profile=st.session_state.user_profile['preferences'], input_embedding=input_emb, liked=True)
                reply, reply_end, st.session_state.user_profile['last_recommendations'] = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'], filter_dict=st.session_state.filter_dict)
                chatbot_reply += reply
                chatbot_reply_end += reply_end
                new_recommendations = st.session_state.user_profile['last_recommendations']

            # If the user gave feedback, find out for which recommended course and update the user profile accordingly
            elif detected_intent == "feedback":
                if len(detected_courses) > 0:
                    #print(f"xxx INTENT FEEDBACK: LEN(DETECTED_COURSES) == {len(detected_courses)}")
                    # Update the user profile with each rated course
                    for (c, sentiment) in detected_courses:
                        st.session_state.user_profile['preferences'] = update_user_profile(st.session_state.user_profile['preferences'], rated_course = (c, 'current'), liked = (sentiment == 'liked'))
                        st.session_state.user_profile['rated_courses'].append(c)
                        #print(f"xxx intent feedback: Updated profile with: {get_current_title(c)} -> {sentiment}")
                    
                    # Generate new recommendations
                    reply, reply_end, st.session_state.user_profile['last_recommendations'] = recommend_courses(user_profile=st.session_state.user_profile['preferences'], rated_courses=st.session_state.user_profile['rated_courses'], previously_liked_courses=st.session_state.user_profile['previously_liked_courses'], filter_dict=st.session_state.filter_dict)
                    chatbot_reply += reply
                    chatbot_reply_end += reply_end
                    new_recommendations = st.session_state.user_profile['last_recommendations']
                

        # Create a button for each recommended title containing all details that should be displayed
        course_buttons = [{"title": course["title"], "details": course} for course in [get_details(c) for c in new_recommendations]]

        # Create the chatbot's reply, append it to the chat history and write it into the chat
        st.session_state.messages.append({"role": "assistant", "content": chatbot_reply, "content_end": chatbot_reply_end, "buttons": course_buttons})
        #print(f"--- CHATBOT_MSG: {st.session_state.messages[-1]}")
        msg_idx = len(st.session_state.messages[-1])-1
        with st.chat_message("assistant"):
            render_message(st.session_state.messages[-1], msg_idx)


###--- Create the buttons for general instructions and hints for feedback at the bottom of the chat ---###
########### AND FOR OTHER HINTS???

# Custom CSS for the box to display the hints in
st.markdown("""
    <style>
        .hint-box {
            border: 1px solid grey;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
            font-size: 10px;
            max-width: 80%
        }
    </style>
""", unsafe_allow_html=True)

# If only the Instructions-Button should be visible
if st.session_state.hint_button['current_hint'] == 'none':
    # Adding the 'Hint' button
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

# If all buttons should be visible ## HARDCODED WITH FEEDBACK HINT
elif st.session_state.hint_button['current_hint'] == 'all':

    # Place the buttons side by side
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Instructions", key="button_1"):
            toggle_hint("show_instructions")
    with col2:
        if st.button("Feedback Hint", key="button_2"):
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
                {hints['Feedback Hint']}
            </div>
        """, unsafe_allow_html=True)


# If two buttons should be visible ## WITH VARIABLE HINT BUTTON
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



###--- Add Sidebar ---###

# Create a sidebar with multiselect widgets
st.sidebar.header("Filter Options", help = "To select a filter, you can either select them from the boxes below or write them in the chat. To remove a filter, just click on the 'x' next to it.")

all_filter = get_all_filter()

# Calling this funtion from the widgets (on_change=update_filters, args=([filter_key, day, times],)) makes it kinda better as then it directly closes after choosing the 1. (but at least saves that selection)
def update_filters(filter):
    print(f"x~x~x~x Updating filter '{filter}'")
    if isinstance(filter, str):
        st.session_state.filter_dict[filter] = st.session_state[filter]
    else:
        st.session_state.filter_dict[filter[0]][filter[1]] = st.session_state[f"{filter[0]}_{filter[1]}_{filter[2]}"]

filter_names = {'status': 'Status', 'mode': 'Mode', 'ects': 'ECTS', 'sws': 'SWS', 'lecturer_short': 'Lecturer', 'module': 'Module', 'language': 'Language', 'filter_time': 'Time'}

for filter_key, filter_options in all_filter.items():
    f_name = filter_names[filter_key]
    f_help = help_text[filter_key] if filter_key in help_text else ""

    # For each day, create a multiselect to choose a timeframe
    if filter_key == 'filter_time':
        #st.sidebar.subheader("Days and Times", help="Attention! Choosing one or more times means that you will only get recommendations for courses that take place in the selected times! If you selected at least one time for at least one day, each other day, for which no times are selected, is filtered out.")
        st.sidebar.subheader("Days and Times", help="Select the days you want to get recommendations for. Due to a bug, you have to click twice for enabling/disabling a day. You can set ranges for a day or multiple days (also ranges allowed) by writing them in the chat. Make sure to deactivate the day before writing the range in the chat, as it would otherwise just be added to the range of the whole day, changing nothing.")
        selected_options = {}
        for day, timeframes in filter_options.items():
            #for times in timeframes:  ### AKTUELL EH NUR JE 1 TIMEFRAME PRO TAG
            #times = timeframes[0]
            #start_time = int(times[0])
            #end_time = int(times[1])
            ### FOR NOW, ONLY ALLOW TO ENABLE/DISABLE WHOLE DAYS (multiselect/slider: siehe backup)
            enable_key = f"allow_day_{day}"
            
            #remove_time = []
            #disable_day = st.checkbox(f"Disable Times for {day}", value=st.session_state[disable_key], help="If all days are disabled, the courses won't be filtered by time. If at least one day is enabled, courses taking place on disabled days will be filtered out.")
            allow_day = st.sidebar.checkbox(f"Show courses on {day}?", value=st.session_state[enable_key], key=f"allow_{day}")
            
            expander = st.sidebar.expander(f"Selected Times for {day}")
            if st.session_state.filter_dict[filter_key][day] != []:
                for day_time in st.session_state.filter_dict[filter_key][day]:
                    if day_time != []:
                        expander.write(f"{day_time[0]}:00 - {day_time[1]}:00")
            else:
                expander.write("")

                    #print(f"DAY_TIME: {day_time}")
                    #if st.button(f"{day_time[0]}:00 - {day_time[1]}:00", key=f"time_{day_time[0]}-{day_time[1]}"):
                    #    remove_time.append(day_time)

                #if st.session_state[f"time_{day_time[0]}-{day_time[1]}"]:
                #    print(f"REMOVED TIME {day_time} from {day}")
                #    selected_options[day].remove(day_time)
            #for t in remove_time:
            #    selected_options[day].remove(t)

            if allow_day:
                st.session_state[enable_key] = True
            else:
                st.session_state[enable_key] = False
                expander.empty()

            if st.session_state[enable_key] == True:
                append_times = []
                if st.session_state.filter_dict[filter_key][day] != []:
                    append_times = st.session_state.filter_dict[filter_key][day]
                else:
                    append_times = [[timeframes[0][0], timeframes[0][1]]]
                if day in selected_options:
                    selected_options[day] += append_times
                else:
                    selected_options[day] = append_times
            else:
                selected_options[day] = []

    # Use a slider with ranges for ects and sws
    elif filter_key in ["ects", "sws"]:
        selected_options = st.sidebar.select_slider(
            label=f"Select {f_name}", 
            options=filter_options,
            value=[filter_options[0], filter_options[-1]],
            key=filter_key
        )

    else:
        # Use st.multiselect and connect it to a session state variable via the `key`
        selected_options = st.sidebar.multiselect(
            label=f"Select {f_name}",
            options=filter_options,
            default=st.session_state.filter_dict[filter_key],  # Initially empty selection
            key=filter_key,  # Unique key for each widget
            help=f_help,
            on_change=update_filters,  # Trigger the filter update funtion
            args=(filter_key,),  # Pass the filter as an argument to the function
        )
    st.session_state.filter_dict[filter_key] = selected_options

print(f"\n~~~ NEW FILTER DICT: {st.session_state.filter_dict}")


