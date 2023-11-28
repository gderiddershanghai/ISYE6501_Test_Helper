import time
from dotenv import load_dotenv, find_dotenv

#====================STATES===================
mid_term_1 = {"input_language": "en", "output_language": "en", "scenario": "wake_word", "quest": "main", "function": get_wake_word}
mid_term_2 = {"input_language": "en", "output_language": "en", "scenario": "intent_detection", "quest": "main", "function": get_intent_turbo}
final_1 = {"input_language": "cs", "output_language": "en", "scenario": "czech_jack", "quest": "main", "function": get_english_translation}

module_review = {"input_language": "en", "output_language": "cs", "scenario": "english_to_czech ", "quest": "main", "function": get_czech_translation}
rephrase = {"input_language": "en", "output_language": "en-cs", "scenario": "rephrase", "quest": "main", "function": correction}
explain_main = {"input_language": "en", "output_language": "en", "scenario": "explain", "quest": "main", "function": explain}

stop = {"input_language": "en", "output_language": "en", "scenario": "stop", "quest": "main", "function": get_intent_turbo}
# __states__ = [wake_word,intent_detection,czech_jack,english_to_czech,rephrase,explain_main,roleplay,roleplay_explain, roleplay_translate_jack, roleplay_translate_to_czech,
# speech, speech_explain,speech_translate_jack,speech_translate_to_czech, book, book_explain, book_translate_jack, book_translate_to_czech, stop]

# #====================STATES_DICTIONARY===================
STATES_DICTIONARY = {
    "book": book,
    "speech": speech,
    "roleplay":roleplay,
    "book": book,
    "roleplay_explain": roleplay_explain,
    "roleplay_translate_jack":roleplay_translate_jack,
    "roleplay_translate_to_czech": roleplay_translate_to_czech,
    "speech_explain": speech_explain,
    "speech_translate_jack":speech_translate_jack,
    "speech_translate_to_czech": speech_translate_to_czech,
    "book_explain": book_explain,
    "book_translate_jack":book_translate_jack,
    "book_translate_to_czech": book_translate_to_czech,
    "intent_detection": intent_detection,
    "other": intent_detection,
    "rephrase": rephrase,
    "czech_translator": english_to_czech,
    "english_explainer": explain_main,
    "speech_writer": speech,
    "english_translator": czech_jack,
    "music": music,
    "music_explain": music_explain,
    "music_translate_jack":music_translate_jack,
    "music_translate_to_czech": music_translate_to_czech,
    }


#====================TOKEN===================
class Token():

    def __init__(self, STATE=wake_word):
        self.STATE = STATE
        self.audio = None
        self.messages = None
        self.user_text = None
        self.system_text = None
        self.message_count = 0
        self.active = True
        self.loops = 0
        self.errors = 0
        self.czech_text = None
        self.side_task_complete = False
        self.roleplay_scenario = None
        self.speak = True
        self.hallucinate = False
        self.interrupted = False

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

    def reset_intent(self):
        print("resetting to intent_detection")
        self.STATE = intent_detection

    def change_state(self, text="intent_detection"):

        if text == "stop":
            self.active = False
        elif text in STATES_DICTIONARY:
            self.STATE = STATES_DICTIONARY[text]
        else:

            text="intent_detection"
            self.STATE = STATES_DICTIONARY[text]


if __name__ == "__main__":
    token = Token(STATE=book)
    print(token.__dict__)
    # token.system_text = "There are seventeen pirates in the basement"
    # token.user_text = "translate the text please"
    # token = book(token)
