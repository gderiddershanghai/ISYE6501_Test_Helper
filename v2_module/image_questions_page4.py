# def generate_image_questions():

#     import streamlit as st
#     from v2_module.states import Token

#     def apply_custom_css():
#         custom_css = """
#         <style>
#             .question-style {
#                 font-size: 18px;

#             }
#             .textbox-style {
#                 font-size: 16px;
#             }
#         </style>
#         """
#         st.markdown(custom_css, unsafe_allow_html=True)

#     def question_generator(label, options, question_key):
#         answer = st.radio(label='Please select the correct answer', options=options, key=question_key)
#         return answer

#     # Check if 'token' is already in the session state, otherwise initialize it
#     if 'token' not in st.session_state:
#         st.session_state.token = Token(STATE='open')
#         st.session_state.token.initialize_image_questions()

#     apply_custom_css()
#     questions = st.session_state.token.mpc_questions[0]
#     image_fp = questions['fp']
#     questions = questions['questions']


#     st.image(image_fp, caption='Chart Question')

#     for i, q in enumerate(questions, start=0):
#         label = q['question']
#         options = q['options_list']
#         correct_answer = q['correct_answer']
#         question_key = f"question_{i}"

#         print('question key', question_key)

#         st.markdown('-------------------------------')
#         st.markdown(f'<div class="question-style">{label}</div>', unsafe_allow_html=True)

#         question = question_generator(label, options, question_key)

#         # Use a unique key for each submit button to handle them individually
#         if st.button('Submit', key=f"submit_{i}"):
#             if question == correct_answer:
#                 st.success('Great work!')
#             else:
#                 st.info(f"The correct answer was {correct_answer}")

def generate_image_questions():

    import streamlit as st
    from v2_module.states import Token

    def apply_custom_css():
        custom_css = """
        <style>
            .question-style {
                font-size: 18px;
            }
            .textbox-style {
                font-size: 16px;
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def question_generator(label, options, question_key):
        answer = st.radio(label='Please select the correct answer', options=options, key=question_key)
        return answer

    # Check if 'token' is already in the session state, otherwise initialize it
    if 'token' not in st.session_state:
        st.session_state.token = Token(STATE='open')
        st.session_state.token.initialize_image_questions()

    apply_custom_css()

    # Safely extract mpc_questions and check structure
    questions_data = st.session_state.token.mpc_questions

    # Check if mpc_questions is structured as expected (list of dictionaries)
    if not isinstance(questions_data, list) or len(questions_data) == 0:
        st.error("No questions available.")
        return

    # Extract the first question set
    questions = questions_data[0]

    # Check if the question set has 'fp' and 'questions' keys
    if 'fp' not in questions or 'questions' not in questions:
        st.error("Question data is not properly formatted.")
        return

    image_fp = questions['fp']  # Filepath to the image
    questions = questions['questions']  # List of questions

    st.image(image_fp, caption='Chart Question')

    # Loop through the list of questions
    for i, q in enumerate(questions):
        # Check if the question has required fields
        print(image_fp, '------------------------')
        if 'question' not in q or 'options_list' not in q or 'correct_answer' not in q:
            st.warning(f"Question {i+1} is missing required fields.")
            continue

        label = q['question']
        options = q['options_list']
        correct_answer = q['correct_answer']
        question_key = f"question_{i}"

        # Display question
        st.markdown('-------------------------------')
        st.markdown(f'<div class="question-style">{label}</div>', unsafe_allow_html=True)

        # Generate the answer options
        selected_answer = question_generator(label, options, question_key)

        # Handle the submit action
        if st.button('Submit', key=f"submit_{i}"):
            if selected_answer == correct_answer:
                st.success('Great work!')
            else:
                st.error(f"The correct answer was {correct_answer}")
