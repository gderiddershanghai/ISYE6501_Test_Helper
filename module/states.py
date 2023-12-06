import time
from dotenv import load_dotenv, find_dotenv
import numpy as np
import random
from general_questions_m1 import MIDTERM_1_QUESTIONS
from general_questions_m2 import MIDTERM_2_QUESTIONS
from general_questions_final import FINAL_QUESTIONS
from knowledge_check_m1 import KNOWLEDGE_1_QUESTIONS
from knowledge_check_m2 import KNOWLEDGE_2_QUESTIONS

from summaries import FINAL_SUMMARY,MIDTERM_1_SUMMARY,MIDTERM_2_SUMMARY
from exam_tester import exam_prep, exam_correction
question_index = np.random.choice(range(25), size=2, replace=False)
q1 = question_index[0]
q2 = question_index[1]

question_index = np.random.choice(range(30), size=2, replace=False)
q1_short = question_index[0]
q2_short = question_index[1]

final_long_form = random.choice([MIDTERM_2_QUESTIONS,MIDTERM_2_QUESTIONS])
final_short_form = random.choice([KNOWLEDGE_1_QUESTIONS,KNOWLEDGE_2_QUESTIONS])

#====================STATES===================
mid_term_1 = {'summary': MIDTERM_1_SUMMARY, 'long_form_question_1': MIDTERM_1_QUESTIONS, 'long_form_question_2': MIDTERM_1_QUESTIONS, 'short_form_question_1': KNOWLEDGE_1_QUESTIONS,'short_form_question_2': KNOWLEDGE_1_QUESTIONS, "function": exam_prep}
mid_term_2 = {'summary': MIDTERM_2_SUMMARY, 'long_form_question_1': MIDTERM_2_QUESTIONS, 'long_form_question_2': MIDTERM_2_QUESTIONS, 'short_form_question_1': KNOWLEDGE_2_QUESTIONS,'short_form_question_2': KNOWLEDGE_1_QUESTIONS, "function": exam_prep}
final = {'summary': FINAL_SUMMARY, 'long_form_question_1': FINAL_QUESTIONS, 'long_form_question_2': FINAL_QUESTIONS, 'short_form_question_1': KNOWLEDGE_1_QUESTIONS,'short_form_question_2': KNOWLEDGE_2_QUESTIONS, "function": exam_prep}
correction = {"function": exam_correction}

module_review = { "function": 1234}

# #====================STATES_DICTIONARY===================
STATES_DICTIONARY = {
    "midterm_1": mid_term_1,
    "midterm_2": mid_term_2,
    "final": final,
    "correction": correction,
    "module_review": module_review,
    }


#====================TOKEN===================
class Token():

    def __init__(self, STATE="final"):
        question_index = np.random.choice(range(25), size=2, replace=False)
        q1 = question_index[0]
        q2 = question_index[1]

        question_index = np.random.choice(range(30), size=2, replace=False)
        q1_short = question_index[0]
        q2_short = question_index[1]

        question_index = np.random.choice(range(50), size=2, replace=False)
        final_1 = question_index[0]
        final_2 = question_index[1]

        self.STATE = STATES_DICTIONARY[STATE]
        self.audio = None
        self.messages = None
        self.user_text = None
        self.system_text = None
        self.message_count = 0
        self.active = True
        self.loops = 0
        self.errors = 0
        self.summary = STATES_DICTIONARY[STATE]['summary']
        if STATE == 'final':
            self.long_form_question_1 = STATES_DICTIONARY[STATE]['long_form_question_1'][final_1]
            self.long_form_question_2 = STATES_DICTIONARY[STATE]['long_form_question_2'][final_2]
        else:
            self.long_form_question_1 = STATES_DICTIONARY[STATE]['long_form_question_1'][q2]
            self.long_form_question_2 = STATES_DICTIONARY[STATE]['long_form_question_2'][q1]
        self.short_form_question_1 = STATES_DICTIONARY[STATE]['short_form_question_1'][q2_short]
        self.short_form_question_2 = STATES_DICTIONARY[STATE]['short_form_question_2'][q1_short]
        self.assistant_message = None
        self.hallucinate = False
        self.interrupted = False
        self.exam_correction = None

    def turn_off(self):
        self.active = False

    def add_message(self, messages):
        if isinstance(messages, list):
            if self.messages is None:
                self.messages = messages
            else:
                self.messages.extend(messages)
        else:
            if self.messages is None:
                self.messages = [messages]
            else:
                self.messages.append(messages)

    def clear_messages(self):
        self.messages = None

    def increase_message_count(self):
        self.message_count += 1

    def add_loop(self):
        self.loops += 1

    def increase_error_count(self):
        self.errors += 1

    def change_state(self, text):
        if text in STATES_DICTIONARY:
            self.STATE = STATES_DICTIONARY[text]



if __name__ == "__main__":
    token = Token(STATE="mid_term_1")
    print(token.__dict__)
    # token.system_text = "There are seventeen pirates in the basement"
    # token.user_text = "translate the text please"
    # token = book(token)
