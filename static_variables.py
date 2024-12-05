## This file contains all dictionaries etc. defining replies, states, ...

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
    ### ADD MORE
}

# Define chatbot replies based on user's intent
intent_replies = {
    "greeting": "Hello! How can I assist you with course recommendations today?",
    "free_description": "Thanks for sharing! ",
    "liked_course_reference": "That's a great course! Let me find some similar courses for you...",
    "feedback": "Thank you for your feedback! I'll use it to improve recommendations.",
    "other": "I didn't understand that. Could you please clarify?"
}

# Define examples for each intent-category
intent_examples = {
    "greeting": ["hello", "hi chatbot", "good morning", "how are you"],
    "free_description": [
        "I want to find a course about machine learning",
        "Can you recommend me an online course?",
        "I am looking for a beginner course in psychology",
        "I want a course from the module artificial intelligence",
        "I want a course from the module Artificial Intelligence",
        "I want a course about philosophy",
        "I want to learn about natural language processing",
        "I want to know more about natural language processing",
        "I like artificial intelligence",
        "I like computer science"
    ],
    "liked_course_reference": [
        "I liked the cognitive neuroscience course",
        "I really liked the course Artificial Intelligence and the Web",
        "I enjoyed Philosophy 101",
        "Neurobiology was interesting",
        "Machine Learning 101 was great",
        "Cognitive neuroscience was nice",
        "I liked Introduction to Artificial Intelligence and Logic Programming",
        "I liked Introduction to epistemology and philosophy of science",
        "I really liked 'Introduction to political philosophy'",
        "I enjoyed Introduction to the Ethics of Artificial Intelligence",
        "Introduction to the Philosophy of Mind was great",
        "Introduction to Cognitive (Neuro-)Psychology was nice",
        "Introduction to Computational Linguistics was good",
        "Introduction to Sleep and Dream was interesting",
        "I want something like Introduction to Unity",
        "I want a course like Introduction to Statistics and Data Analysis",
        "I'm looking for a seminar like Neuroscience and Philosophy",
        "I want a lecture similar to Introduction to Logic and Critical Thinking"
    ],
    "feedback": [
        "I liked the second course", 
        "The recommendations were not good", 
        "I want more courses like the first one", 
        "I want more courses like the second one", 
        "I want more courses like number three",
        "I like the first recommendation",
        "I like the last one",
        "The third one sounds interesting",
        "Number three sounds good",
        "I like course 4",
        "The fifth one is interesting",
        "I only like number 3",
        "I like the course 2",
        "Course 1 sounds good",
        "Course three is interesting",
        "The final recommendation is interesting",
        "I like the first, third, fourth and fifth one",
        "I like number one, two and five",
        "I like course number 2, 4 and 5",
        "I don't like the first one",
        "Number 2 does not sound interesting",
        "Course 5 is not interesting",
        "Course two sounds not good",
        "The last course doesn't sound good",
        "Your final recommendation does not sound good",
        "The last one is not good",
        "The last course is bad",
        "I dislike the first recommendation",
        "I dislike course 4",
        "I dislike course three",
        "Number five sounds boring",
        "Course 4 sounds boring",
        "Course one, two and three sound boring",
        "Number 1 and 5 sound boring",
        "I don't like 3, 4 and 2",
        "I hate number 3",
        "I hate course two",
        "The first and second one sound good, but I don't like the fifth one",
        "I like all of them",
        "All sound interesting",
        "All of them sound good",
        "I like them all",
        "I like all recommendations",
        "I like every recommended course",
        "Every recommendation is good",
        "I don't like any of them",
        "None of them seem good",
        "None of them are interesting",
        "I like none of them",
        "I like none of the courses",
        "I like none of the recommendations",
        "I hate all of them",
        "I like 1 and 3, but 2, 4 and 5 are boring",
        "Three and four are interesting. Two and five are",
        "I like course five but not four",
        "I dislike number 3 and 5, but not 2",
        "Course three and 4 are good, but 2 is boring",
        "I like the first course. But not the second",
        "I like 1, 3, 5, but not 2 and four"
        ]
}

instructions = {
    "feedback": "When giving feedback, please refer to the course you want to give feedback on by it's position in the list (e.g., 'the first' or 'course 4'). Please write each course separately instead of giving ranges like 'courses 2 to 4'. You can, however, tell me if you liked all or none of the recommendations (e.g., 'I like all of them'). Also, if you want to give both positive and negative feedback for some recommended courses in a single message, please make sure that you don't write them in the same sentence or that you divide the sentence using 'but'. For example, you could write: 'I liked the first and third recommendation. But I didn't like the second one.'"
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
            </ul>
        </div>""",
    'How to start': 
        """<div style="padding-left: 10px; max-width: 80%; font-size: 10px;">
        <p>Getting started!</p>
        </div>""",
    'Feedback Hint': "How to give feedback! Lets see "

}
