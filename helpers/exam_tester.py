import os
import openai
from dotenv import load_dotenv, find_dotenv
import re

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def exam_prep(token):
    first_system_message = """
    You are a study guide assistant for a student preparing for the final test in Georgia Tech's ISYE 6501 course. Your role is to help the student review course materials and practice with exam-style questions. Please follow the user's instructions and assist them in the best way possible.
    """

    first_user_message = """
    1. Read the course summary
    2. Read the example question from a previous exam
    3. Generate a question in the style of the previous exam that checks whether the student has understood the material. It can use information from the summary as well as other information that the model has.
    The role of the model will be a study guide to prepare students for their Georgia Tech's ISYE 6501 final test
    """
    if token.loops == 0:
        messages = [{'role':'system', 'content': first_system_message},
                    {'role':'user', 'content': token.user_text}]
    else: system_message = """
    You are a study guide assistant for a student preparing for the final test in Georgia Tech's ISYE 6501 course. Your role is to help the student review course materials by checking his/her answers to exam questions.
    """


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=150
    )
    answer = response.choices[0].message["content"]
    print(answer)

    try:
        correction, explanation = answer.split("\n\n")
        token.system_text = correction
        token.czech_text = explanation

    except Exception as e:
        print("Error with the format: {0}".format(e))
        token.increase_error_count()
        if "task" in answer.lower():
            try:
                task2_pattern = re.compile(r"TASK 2: Rephrase the target sentence\.(.*)TASK 3: Explain in Czech why the original English sentence was incorrect\.(.*)", re.DOTALL)
                match = task2_pattern.search(answer)

                if match:
                    token.system_text = match.group(1).strip()
                    token.czech_text = match.group(2).strip()
            except:
                print("No taks in answer: {0}".format(e))
                token.increase_error_count()
                token.system_text = "Please try again"
                token.czech_text = ""

    if token.speak: speak(token)
    token.change_state("intent_detection")

    return token


if __name__ == "__main__":
    print("test")
