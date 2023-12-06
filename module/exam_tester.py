import os
import openai
from dotenv import load_dotenv, find_dotenv
import re

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# def exam_prep(token):

#     first_system_message = f"""
#     You are an advanced AI designed to assist in educational contexts, specifically tailored for the ISYE 6501 course at Georgia Tech. Your capabilities include generating new, unique, and challenging questions for exam preparation. You have access to course outlines, previous exam questions, and general knowledge about the subject matter.

#     The student is currently studying the following topics covered in the course:

#     OUTLINE
#     {token.summary}

#     You are provided with examples of the format and style of questions that have appeared in past exams:

#     EXAMPLE 1
#     {token.long_form_question_1}
#     EXAMPLE 2
#     {token.short_form_question_2}

#     In generating questions, please adhere to the following guidelines:
#     - Emphasize creating complex, scenario-based questions that require applying knowledge from the course in practical situations.
#     - Ensure the questions involve multiple steps of reasoning or analysis, similar to the detailed example provided.
#     - Innovate within the constraints of the course material, making sure the difficulty level is on par with the final exam.
#     - Avoid replicating previous questions verbatim; instead, use them as a stylistic and complexity reference.
#     - The questions should comprehensively cover the key topics outlined, demonstrating a deep understanding of the material.
#     - Format the questions clearly, specifying the type (MPC, T/F, or matching) and providing choices or statements as needed.
#     - Create four distinct questions, each reflecting a real-world application or a complex scenario related to the course content.

#     Your task is to generate questions that assess the student's comprehension and retention of the course material in a nuanced and in-depth manner. These questions should be challenging and thought-provoking, preparing the student for the high level of analytical thinking required in the final exam.
#     """

#     first_user_message = f"""
#     As a student preparing for the ISYE 6501 final exam, I am looking for challenging, in-depth practice questions. Based on my course study and previous exam formats, please generate questions that test a comprehensive understanding of the material. Here's what I need:

#     - Four questions in total, each presenting a unique and thought-provoking scenario.
#     - Focus on complex, multi-step problems that require analytical thinking and application of course concepts.
#     - For multiple-choice questions, provide four options (A, B, C, and D) and ensure they are scenario-based, requiring detailed analysis to solve.
#     - True/False questions should present scenarios where I need to apply my knowledge to determine the accuracy of a statement.
#     - If you include matching questions, they should involve matching complex concepts or scenarios rather than straightforward definitions.
#     - The questions should not only test factual knowledge but also the application of concepts in real-world or hypothetical scenarios.
#     - Avoid repeating questions from previous exams but feel free to use their complexity and depth as a benchmark.

#     These questions should mimic the style and challenge of the actual final exam, helping me to thoroughly prepare for it. Thank you for your assistance in creating these practice questions!
#     """

#     messages = [{'role':'system', 'content': first_system_message},
#                 {'role':'user', 'content': first_user_message}]

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0.9,
#         max_tokens=577
#     )
#     answer = response.choices[0].message["content"]
#     print('----------')
#     print(answer)
#     token.exam_questions = answer

#     return token

def exam_prep(token):



    question = f"""
    {token.long_form_question_1}
    ___________________________________________________________________________

    {token.short_form_question_2}

    ___________________________________________________________________________
    {token.short_form_question_1}
    """


    token.exam_questions = question

    return token




def exam_correction(token):
    print('entering exam correction')
    system_message = """
    You are a study guide assistant for a student preparing for the final test in Georgia Tech's ISYE 6501 course.
    Your role is to correct answers to exam questions and if the student's asnwer isn't correct, to explain why.
    """
    messages = [{'role':'system', 'content': system_message},
                {'role':'user', 'content': 'Please ask me some questions to help me prepare for my test.'},
                {'role': 'assistant', 'content': token.exam_questions},
                {'role':'user', 'content': token.user_text}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            max_tokens=500
        )
    answer = response.choices[0].message["content"]
    print('>>>>>>>CORRECTIONS>>>>>>>>')
    print(answer)
    print('>>>>>>>CORRECTIONS>>>>>>>>')
    token.exam_correction = answer

    return token

if __name__ == "__main__":
    print("test")
