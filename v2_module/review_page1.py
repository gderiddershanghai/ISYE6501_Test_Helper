
def sa_questions():

    import streamlit as st
    from v2_module.states import Token

    def apply_custom_css():
        custom_css = """
        <style>
            .question-style {
                font-size: 20px; /* Customize the font size as needed */
                font-weight: bold; /* Optional: Make the question bold */
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def question_generator(label, options, question_key):
        question = st.radio(label='Please select the correct answer', options=options, key=question_key)
        return question

    if 'token' not in st.session_state:
        st.session_state.token = Token(STATE='review')
        st.session_state.token.initialize_mpc_questions()

    apply_custom_css()
    questions = st.session_state.token.mpc_questions

    # for i, q in enumerate(questions, start=0):
    #     label = q['question']
    #     options = q['options_list']
    #     correct_answer = q['correct_answer']
    #     question_key = f"question_{i}"
    #     explanation = q['explanation']

    #     print('question key', question_key)

    #     st.markdown('-------------------------------')
    #     st.markdown(f'<div class="question-style">{label}</div>', unsafe_allow_html=True)

    #     question = question_generator(label, options, question_key)

    #     if st.button('Submit', key=f"submit_{i}"):
    #         if question == correct_answer:
    #             st.success('Great work!')
    #             st.info(f'Explanation: \n\n{explanation}')
    #         else:
    #             st.error(f"The correct answer was {correct_answer}")
    #             st.info(f'Explanation: \n\n{explanation}')

    #         if 'chapter_information' in q:
    #             st.write(f"You can review {q['chapter_information']}")
    for i, q in enumerate(questions, start=0):
                label = q['question']
                options = q['options_list']
                correct_answer = q['correct_answer']
                question_key = f"question_{i}"
                explanation = q['explanation']

                st.markdown('-------------------------------')
                # Use st.markdown to render the question, allowing LaTeX and markdown formatting
                st.markdown(f"**{label}**")

                question = question_generator(label, options, question_key)

                # Store submitted answers in session state
                if st.button('Submit', key=f"submit_{i}"):
                    if 'submitted_answers' not in st.session_state:
                        st.session_state.submitted_answers = {}

                    st.session_state.submitted_answers[question_key] = question
                    print('question', question)
                    print(correct_answer, 'corect')
                    if question not in ['True', 'False']: question = question[0]
                

                    if question == correct_answer:
                        st.success('Great work!')
                        st.info(f'Explanation: \n\n{explanation}')
                    else:
                        st.error(f"The correct answer was {correct_answer}")
                        st.info(f'Explanation: \n\n{explanation}')

                    if 'chapter_information' in q:
                        st.write(f"You can review {q['chapter_information']}")



# if __name__ == "__main__":
#     main()
