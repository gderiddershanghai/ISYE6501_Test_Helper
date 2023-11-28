
import openai
import os
from dotenv import load_dotenv, find_dotenv
import time
from helpers.states import Token



# Add evironment variables, incl. OpenAI API key and GCP credentials
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')


def chatbot():
    token = Token()
    token.speak = False #for testing, so we don't get charged for synthesis
    # Continuously listen for user input
    input_cycle_count = 0
    while token.active == True:
        input_cycle_count += 1

        time.sleep(0.1)
        token.user_text = input("Enter your input: ")

        token = token.STATE.get("function")(token)
        if token.messages is not None:
            for i, message in enumerate(token.messages):

        else:
            print('123')


if __name__ == "__main__":
    chatbot()
