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

    # Check if 'token' is in session state, otherwise initialize it
    if 'token' not in st.session_state:
        st.session_state.token = Token(STATE='chart')
        st.session_state.token.initialize_image_questions()

    apply_custom_css()

    # Extract image questions from the session state
    questions_data = st.session_state.token.image_questions

    # Validate the structure of image questions
    if not isinstance(questions_data, list) or len(questions_data) == 0:
        st.error("No image questions available.")
        return

    # Access the first set of questions
    questions = questions_data[0]

    # Check that the required keys exist
    if 'fp' not in questions or 'questions' not in questions:
        st.error("Question data is not properly formatted.")
        return

    # Display the image
    image_fp = questions['fp']  # Filepath to the image
    st.image(image_fp, caption='Chart Question')

    # Loop through and display each question
    for i, q in enumerate(questions['questions']):
        if 'question' not in q or 'options_list' not in q or 'correct_answer' not in q:
            st.warning(f"Question {i+1} is missing required fields.")
            continue

        label = q['question']
        options = q['options_list']
        correct_answer = q['correct_answer']
        question_key = f"question_{i}"

        # Display question and answer options
        st.markdown('-------------------------------')
        st.markdown(f'<div class="question-style">{label}</div>', unsafe_allow_html=True)

        selected_answer = question_generator(label, options, question_key)

        # Handle the submit action for each question
        if st.button('Submit', key=f"submit_{i}"):
            if selected_answer == correct_answer:
                st.success('Great work!')
            else:
                st.error(f"The correct answer was {correct_answer}")
