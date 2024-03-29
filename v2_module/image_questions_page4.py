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
    questions = st.session_state.token.mpc_questions[0]
    image_fp = questions['fp']
    questions = questions['questions']


    st.image(image_fp, caption='Chart Question')

    for i, q in enumerate(questions, start=0):
        label = q['question']
        options = q['options_list']
        correct_answer = q['correct_answer']
        question_key = f"question_{i}"

        print('question key', question_key)

        st.markdown('-------------------------------')
        st.markdown(f'<div class="question-style">{label}</div>', unsafe_allow_html=True)

        question = question_generator(label, options, question_key)

        # Use a unique key for each submit button to handle them individually
        if st.button('Submit', key=f"submit_{i}"):
            if question == correct_answer:
                st.success('Great work!')
            else:
                st.info(f"The correct answer was {correct_answer}")
