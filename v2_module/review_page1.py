
# def sa_questions():

#     import streamlit as st
#     from v2_module.states import Token

#     def apply_custom_css():
#         custom_css = """
#         <style>
#             .question-style {
#                 font-size: 20px; /* Customize the font size as needed */
#                 font-weight: bold; /* Optional: Make the question bold */
#             }
#         </style>
#         """
#         st.markdown(custom_css, unsafe_allow_html=True)

#     def question_generator(label, options, question_key):
#         question = st.radio(label='Please select the correct answer', options=options, key=question_key)
#         return question

#     if 'token' not in st.session_state:
#         st.session_state.token = Token(STATE='review')
#         st.session_state.token.initialize_mpc_questions()

#     apply_custom_css()
#     questions = st.session_state.token.mpc_questions

#     for i, q in enumerate(questions, start=0):
#         label = q['question']
#         if 'options_list' in q:
#             options = q['options_list']
#         else:
#             # Handle the missing key scenario (e.g., log an error or provide a default value)
#             options = []
#             print("Warning: 'options_list' key is missing from the dictionary")
#         correct_answer = q['correct_answer']
#         question_key = f"question_{i}"
#         explanation = q['explanation']

#         print('question key', question_key)

#         st.markdown('-------------------------------')
#         st.markdown(label)

#         question = question_generator(label, options, question_key)

#         if st.button('Submit', key=f"submit_{i}"):
#             if question == correct_answer:
#                 st.success('Great work!')
#                 st.info(f'Explanation: \n\n{explanation}')
#             else:
#                 st.error(f"The correct answer was {correct_answer}")
#                 st.info(f'Explanation: \n\n{explanation}')

#             if 'chapter_information' in q:
#                 st.write(f"You can review {q['chapter_information']}")
   


def sa_questions():
    import streamlit as st
    from v2_module.states import Token

    def apply_custom_css():
        custom_css = """
        <style>
            .question-style {
                font-size: 20px;
                font-weight: bold;
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def question_generator(label, options, question_key):
        question = st.radio(label='Please select the correct answer', options=options, key=question_key)
        return question

    # Select exam type (midterm1, midterm2, final)
    exam_type = st.selectbox("Select Exam Type", ["Select an option", "midterm1", "midterm2", "final"])

    # Initialize token only if it does not exist
    if 'token' not in st.session_state:
        st.session_state.token = Token()

    # Reinitialize questions when a new exam type is selected
    if exam_type != "Select an option" and st.session_state.token.STATE != exam_type:
        # Set the exam type in token state and reinitialize questions
        st.session_state.token.STATE = exam_type
        st.session_state.token.mpc_questions = []  # Clear previous questions
        st.session_state.token.initialize_mpc_questions(exam_type)

    # Apply custom CSS styling
    apply_custom_css()

    # Display questions only if exam type has been selected and questions are loaded
    questions = st.session_state.token.mpc_questions
    if questions:
        for i, q in enumerate(questions):
            label = q['question']
            options = q.get('options_list', [])
            correct_answer = q['correct_answer']
            question_key = f"question_{i}"
            explanation = q['explanation']

            st.markdown('-------------------------------')
            st.markdown(label)
            question = question_generator(label, options, question_key)

            if st.button('Submit', key=f"submit_{i}"):
                if question == correct_answer:
                    st.success('Great work!')
                    st.info(f'Explanation: \n\n{explanation}')
                else:
                    st.error(f"The correct answer was {correct_answer}")
                    st.info(f'Explanation: \n\n{explanation}')
                if 'chapter_information' in q:
                    st.write(f"You can review {q['chapter_information']}")
    else:
        st.write("Please select an exam type to display questions.")
