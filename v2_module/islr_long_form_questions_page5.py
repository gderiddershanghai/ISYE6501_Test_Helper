# def generate_open_questions():
#     import streamlit as st
#     from v2_module.states import Token

#     # Define custom CSS for question and answer styling
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
#             .streamlit-expanderHeader {
#                 font-size: 16px;
#                 font-weight: bold;
#             }
#             .answer-container {
#                 padding: 10px;
#                 background-color: #e8e8e8;
#                 border-radius: 5px;
#                 margin-bottom: 10px;
#             }
#             .answer-header {
#                 font-weight: bold;
#                 margin-bottom: 5px;
#             }
#             .answer-text {
#                 margin-bottom: 5px;
#             }
#         </style>
#         """, unsafe_allow_html=True)

#     # Dropdown for selecting exam type
#     exam_type = st.selectbox("Select Exam Type", ["Select an option", "midterm1", "midterm2", "final"])

#     # Check if a valid exam type has been selected before initializing questions
#     if exam_type != "Select an option":
#         # Initialize token and load questions based on the selected exam type
#         if 'token' not in st.session_state:
#             st.session_state.token = Token(STATE='open')

#         # Only initialize questions if the exam type has changed or if it's the first load
#         if st.session_state.token.STATE != exam_type:
#             st.session_state.token.STATE = exam_type
#             st.session_state.token.open_questions = []  # Clear previous questions
#             st.session_state.token.initialize_open_questions(exam_type)

#         # Get questions from the session state
#         questions = st.session_state.token.open_questions


#         # Display questions in a styled format
#         for i, q in enumerate(questions, start=1):
#             if "question" in q:
#                 st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
#             else:
#                 print("Warning: 'question' key is missing in q", q)
#             with st.container():
#                 st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
#                 st.markdown(f'<div class="question-header">Question {i}</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
#                 answer = st.text_area("", key=f"question_{i}", placeholder="Type your answer here...", height=150)

#                 if st.button('Compare Answers', key=f"submit_{i}"):
#                     # Correct answer container
#                     st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
#                     st.markdown(f'<div class="answer-header">Correct Answer:</div>', unsafe_allow_html=True)
#                     st.markdown(f'<div class="answer-text">{q["correct_answer"]}</div>', unsafe_allow_html=True)
#                     st.markdown(f'</div>', unsafe_allow_html=True)

#                     # Explanation container
#                     st.markdown(f'<div class="answer-container">', unsafe_allow_html=True)
#                     st.markdown(f'<div class="answer-header">Explanation:</div>', unsafe_allow_html=True)
#                     st.markdown(f'<div class="answer-text">{q["explanation"]}</div>', unsafe_allow_html=True)
#                     st.markdown(f'</div>', unsafe_allow_html=True)
                
#                 st.markdown('</div>', unsafe_allow_html=True)
#     else:
#         st.info("Please select an exam type to display questions.")
def generate_open_questions():
    import streamlit as st
    from v2_module.states import Token

    # Dropdown for selecting exam type
    exam_type = st.selectbox("Select Exam Type", ["Select an option", "midterm1", "midterm2", "final"])

    # Check if a valid exam type has been selected before initializing questions
    if exam_type != "Select an option":
        # Initialize token and load questions based on the selected exam type
        if 'token' not in st.session_state:
            st.session_state.token = Token(STATE='open')

        # Only initialize questions if the exam type has changed or if it's the first load
        if st.session_state.token.STATE != exam_type:
            st.session_state.token.STATE = exam_type
            st.session_state.token.open_questions = []  # Clear previous questions
            st.session_state.token.initialize_open_questions(exam_type)

        # Get questions from the session state
        questions = st.session_state.token.open_questions

        # Display questions in Markdown format
        for i, q in enumerate(questions, start=1):
            st.markdown(f"### Question {i}")
            if "question" in q:
                st.markdown(q["question"])

            # Answer input field
            answer = st.text_area(f"Your Answer (Question {i})", key=f"question_{i}", placeholder="Type your answer here...", height=150)

            # Button to compare answers
            if st.button(f'Compare Answer for Question {i}', key=f"submit_{i}"):
                # Display correct answer and explanation
                st.markdown("#### Correct Answer")
                st.markdown(q["correct_answer"])
                
                st.markdown("#### Explanation")
                st.markdown(q["explanation"])

            st.markdown("---")  # Add a horizontal line between questions
    else:
        st.info("Please select an exam type to display questions.")
