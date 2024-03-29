# def generate_open_questions():
#     import streamlit as st
#     from v2_module.states import Token

#     st.markdown("""
#         <style>
#             .question-container {
#                 padding: 10px;
#                 border-radius: 5px;
#                 margin: 10px 0;
#                 border: 1px solid #EEE;
#                 background-color: #FAFAFA;
#             }
#             .question-header {
#                 font-weight: bold;
#                 color: #333;
#                 margin-bottom: 5px;
#             }
#             .question-text {
#                 font-size: 16px;
#                 margin-bottom: 10px;
#             }
#             .compare-button {
#                 background-color: #FF4B4B;
#                 color: white;
#                 padding: 8px 24px;
#                 border: none;
#                 border-radius: 4px;
#                 margin-top: 10px;
#             }
#             .compare-button:hover {
#                 background-color: #FF6868;
#             }
#             .streamlit-expanderHeader {
#                 font-size: 16px;
#                 font-weight: bold;
#             }
#         </style>
#         """, unsafe_allow_html=True)

#     if 'token' not in st.session_state:
#         st.session_state.token = Token(STATE='open')
#         st.session_state.token.initialize_open_questions()

#     questions = st.session_state.token.mpc_questions

#     for i, q in enumerate(questions, start=1):
#         with st.container():
#             st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
#             st.markdown(f'<div class="question-header">Question {i}</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
#             correct_answer = q['correct_answer']
#             explanation = q['explanation']
#             answer = st.text_area("", key=f"question_{i}", placeholder="Type your answer here...", height=150)
#             # if st.button('Compare Answers', key=f"compare_{i}"):
#             #     with st.expander("See Correct Answer"):
#             #         st.write(q["correct_answer"])
#             if st.button('Compare Answers', key=f"submit_{i}"):
#                 st.info(f"{correct_answer}")
#                 st.info(f'Explanation: \n\n{explanation}')
#             st.markdown('</div>', unsafe_allow_html=True)
def generate_open_questions():
    import streamlit as st
    from v2_module.states import Token

    st.markdown("""
        <style>
            .question-container {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                border: 1px solid #EEE;
                background-color: #FAFAFA;
            }
            .question-header {
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .question-text {
                font-size: 16px;
                margin-bottom: 10px;
            }
            .streamlit-expanderHeader {
                font-size: 16px;
                font-weight: bold;
            }
            .answer-container {
                padding: 10px;
                background-color: #e8e8e8;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .answer-header {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .answer-text {
                margin-bottom: 5px;
            }
        </style>
        """, unsafe_allow_html=True)

    if 'token' not in st.session_state:
        st.session_state.token = Token(STATE='open')
        st.session_state.token.initialize_open_questions()

    questions = st.session_state.token.mpc_questions

    # for i, q in enumerate(questions, start=1):
    #     with st.container():
    #         st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
    #         st.markdown(f'<div class="question-header">Question {i}</div>', unsafe_allow_html=True)
    #         st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
    #         answer = st.text_area("", key=f"question_{i}", placeholder="Type your answer here...", height=150)
    #         if st.button('Compare Answers', key=f"submit_{i}"):
    #             st.markdown('#### Correct Answer:')
    #             st.info(q["correct_answer"])
    #             st.markdown('#### Explanation:')
    #             st.info(q["explanation"])
    #         st.markdown('</div>', unsafe_allow_html=True)
    for i, q in enumerate(questions, start=1):
        with st.container():
            st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="question-header">Question {i}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
            answer = st.text_area("", key=f"question_{i}", placeholder="Type your answer here...", height=150)
            if st.button('Compare Answers', key=f"submit_{i}"):
                # Wrap the answer in a container with a specific class for styling
                st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-header">Correct Answer:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-text">{q["correct_answer"]}</div>', unsafe_allow_html=True)
                st.markdown(f'</div>', unsafe_allow_html=True)

                # Wrap the explanation in a container with a specific class for styling
                st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-header">Explanation:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-text">{q["explanation"]}</div>', unsafe_allow_html=True)
                st.markdown(f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
