import streamlit as st
from v2_module.review_page1 import sa_questions as generate_mpc_questions
from v2_module.islr_long_form_questions_page5 import generate_open_questions
from  v2_module.image_questions_page4 import generate_image_questions as generate_chart_questions

def intro():
    st.write("# Welcome to the ISYE 6501 Midterm 1 Prep App! 👋")
    st.sidebar.success("Select a practice category to begin.")

    st.markdown("""
This interactive tool is designed to aid students in preparing for the ISYE 6501 Midterm 1 exam. Use this app to work through a variety of questions that cover course material effectively. The app includes:

1. **MPC Questions**: Enhance your understanding with knowledge check questions from the course and additional questions generated by GPT-4.
2. **Open-Ended Questions**: Practice with complex, multi-part questions similar to what you might encounter on the exam. These questions are designed by providing GPT-4 with detailed prompts and course content.
3. **Chart & Graph Interpretation**: Improve your ability to read and interpret charts and graphs with questions currently being developed.

### Additional Resources:
- [Lecture Transcripts](https://docs.google.com/document/d/1Hxi5GWcZd3GSm7V6cdnm1HgMvN93eyAaWyX-V3mlRik/edit#heading=h.9af8llygue3h)
- [Course Website: ISYE 6501 - Intro to Analytics Modeling](https://omscs.gatech.edu/isye-6501-intro-analytics-modeling)

If you find this app useful, please consider giving it a star on GitHub: [ISYE6501_Test_Helper](https://github.com/gderiddershanghai/ISYE6501_Test_Helper). Your support helps us improve and expand the features!

Or check out my other review apps for:
- [ISYE 6501 Introduction to Analytics Modeling](https://isye6501test-prep.streamlit.app/)
- [MGT 6203 Analytics for Business](https://mgt-6203-mt-study-aid.streamlit.app/)
- [ISYE 6740 Computational Data Analytics](https://cda-review-app.streamlit.app/)
- [CS 7643 Deep Learning](https://deep-learning-practice.streamlit.app/)
- [CS 7280 Network Science Review App](https://network-science-review.streamlit.app/)


    """)

def mpc_questions():
    st.markdown("# MPC Questions")
    st.write('Knowledge Check and Additional Questions')
    generate_mpc_questions()

def open_ended_questions():
    st.markdown("# Open-Ended Questions")
    st.write('Complex, Multi-part Questions')
    generate_open_questions()

def chart_questions():
    st.markdown("# Chart & Graph Interpretation")
    st.write('Data Interpretation Questions')
    generate_chart_questions()

def reset_or_initialize_state():
    keys_to_delete = ['token', 'mpc_state', 'open_ended_state', 'chart_state']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

page_names_to_funcs = {
    "—": intro,
    "MPC Questions": mpc_questions,
    "Open Ended": open_ended_questions,
    "Charts & Graphs": chart_questions
}

if 'current_demo' not in st.session_state:
    st.session_state['current_demo'] = None

# Sidebar selection box
demo_name = st.sidebar.selectbox("Choose Practice Type", list(page_names_to_funcs.keys()), index=0)

# Check if the demo has changed
if st.session_state['current_demo'] != demo_name:
    st.session_state['current_demo'] = demo_name  # Update the current demo
    reset_or_initialize_state()  # Reset or initialize state based on new demo

# Dynamically call the selected demo function
if demo_name in page_names_to_funcs:
    page_names_to_funcs[demo_name]()
