# Define which user responses can mean yes and which can mean no
confirmation_dict = {
    'yes': ['yes', 'exactly', 'correct', 'yep', 'yeah', 'ja', 'right'],
    'no': ['no', 'nope', 'not', 'false', 'nein', 'wrong']
}

# Define how the chatbot replies to confirmation or rejection
confirmation_replies = {
    'yes': 'Great! ',
    'no': "I'm sorry for the misunderstanding. Could you rephrase your message then? ",
    'other': "I'm sorry, I didn't understand that. Please just answer with \"yes\" or \"no\". We will continue finding great courses after that. "
}

# Define common abbreviations and their full forms
abbreviations = {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "NLP": "Natural Language Processing",
    "NLU": "Natural Language Understanding",
    "CL": "Computational Linguistics",
    "CV": "Computer Vision",
    "Coxi": "Cognitive Science",
    "CogSci": "Cognitive Science",
    "Philo": "Philosophy",
    "Intro": "Introduction",
    "Scipy": "Scientific Programming in Python",
    "Info": "Informatics",
    "Neuroinfo": "Neuroinformatics",
    "Math": "Mathematics",
    "Neurobio": "Neurobiology",
    "HCI": "Human-Computer-Interactions",
    "MCI": "Mensch-Computer-Interaktion",
    "DL": "deep learning",
    "VR": "virtual reality"
    ### ADD MORE???
}

# Define chatbot replies based on user's intent
intent_replies = {
    "greeting": "Hello! How can I assist you with course recommendations today?",
    "free_description": "Thanks for sharing!  \n",
    "liked_course_reference": "That's a great course! Let me find some similar courses for you...",
    "feedback": "Thank you for your feedback! I'll use it to improve recommendations.",
    "nonsense": "I'm sorry, but your message does not make sense to me. I can only understand messages in the context of recommending university courses. For more instructions click on the 'Instructions' button below the chat. Could you please rephrase your message to be more clear?",
    "other": "I'm sorry, but I do not understand what you want to tell me. Could you please clarify?"
}

# Title and text for the hint-button
hints = {
    'instructions':
        """<div style="padding-left: 10px; max-width: 80%; font-size: 10px;">
        <p>You can either tell me a course you liked in the past or simply describe what kind of course you are looking for. Please keep in mind the following tips:</p>
            <ul>
                <li>Keep it simple! It's better to write multiple simple messages than to put all the information into a single one.</li>
                <li>Please don't put different input (such as a reference to a course you liked in the past, a free description of what you would like, or feedback on a recommended course) into a single message. Instead, split it up into multiple messages so that I can better understand you.</li>
                <li>If you want to tell me more than one course you liked in the past, please do so in separate messages (1 course per message).</li>
                <li>Sometimes, I have a hard time understanding the intention of a message. If I don't understand you correctly, you can just start your message by typing out your intention. 
                    <ul>
                        <li>To give feedback, start with 'Feedback:'</li>
                        <li>To tell me a course you liked in the past, start with 'Ref:'</li>
                        <li>To give a free description, start with 'Free:'.</li>
                    </ul>
                </li>
            </ul>
        </div>""",
    'How to start': 
        """<div style="padding-left: 10px; max-width: 80%; font-size: 10px;">
        <p>Getting started!</p>
        </div>""",
    'Feedback Hint': 
        """<div style="padding-left: 10px; max-width: 80%; font-size: 10px;">
            <ul>
                <li>When giving feedback, please refer to the course you want to give feedback to by it's position in the list (e.g., <i>'the first'</i> or <i>'course 4'</i>).</li>
                <li>Please write each course separately instead of giving ranges like <i>'courses 2 to 4'</i>. You can, however, tell me if you liked <i>all</i> or <i>none</i> of the recommendations (e.g., <i>'I like all of them'</i>).</li>
                <li>Also, if you want to give both positive and negative feedback for some recommended courses in a single message, please make sure that you don't write them in the same sentence or that you divide the sentence using <i>'but'</i>. For example, you could write: <i>'I liked the first and third recommendation, but I didn't like the second one.'</i></li>
            </ul>
            </div>"""
}

# Text to add to certain widgets to help the user 
help_text = {
    'module': "The naming of the modules consists of the following information:  \n- 'CS-': Cognitive Science (currently only courses from Cognitive Science modules are included)  \n- 'B'|'M': 'B' for modules from the Bachelor's program, 'M' for the Master's program  \n- 'P'|'W'|'WP': The type of the module - 'P' for compulsory modules, 'WP' for elective modules, 'W' for Distinguishing Elective Courses and Instruction for Working Scientifically  \n- '-XY': The short form of the area (e.g., 'AI' for Artificial Intelligence)  \n\nExample: The module 'CS-BWP-NI' is the elective Neuroinformatics module for the Bachelor's program"

}

# To find mentioned modules
module_dict = {
    'study_program': {
        'bachelor': 'B',
        'master': 'M'
    },
    'module': {
        'elective': 'WP',
        'compulsory': 'P'
        # 'BW' is only used for 'Instruction for Working Scientifically' ('Anleitung zum wissenschaftlichen Arbeiten') -> not important here
    },
    'area': {
        'ai': '-AI',
        'artificial intelligence': '-AI',
        'ni': '-NI',
        'neuroinformatics': '-NI', 
        'neuroinfo': '-NI',
        'cl': '-CL', 
        'linguistics': '-CL',
        'cnp': '-CNP', 
        'neuropsychology': '-CNP', 
        'psychology': '-CNP',
        'mat': '-MAT',
        'mathematics': '-MAT', 
        'math': '-MAT',
        'phil': '-PHIL', 
        'philosophy': '-PHIL', 
        'philo': '-PHIL',
        'working scientifically': '-IWS', 
        'iws': '-IWS',
        'methods': '-MCS',
        'mcs': '-MCS',
        'ic': '-IC', 
        'interdisciplinary': '-IC',
        'inf': '-INF', 
        'info': '-INF', 
        'informatics': '-INF',
        'computer science': '-INF',
        'study project': '-SP', 
        'sp': '-SP',
        'ns': '-NS', 
        'neuroscience': '-NS',
        'distinguishing elective': ''
    }
}