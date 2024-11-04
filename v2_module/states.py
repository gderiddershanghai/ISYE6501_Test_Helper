import numpy as np
# from v2_module.v2_knowledge_check import KC_MPC_QUESTIONS
# from v2_module.v2_long_form import OPEN_QUESTIONS
from v2_module.chart_math_questions import IMAGE_QUESTIONS

from v2_module.midterm1_mpc import MIDTERM1_MPC_QUESTIONS
from v2_module.midterm2_mpc import MIDTERM2_MPC_QUESTIONS

from v2_module.midterm1_open import MIDTERM1_OPEN_QUESTIONS
from v2_module.midterm2_open import MIDTERM2_OPEN_QUESTIONS

# #====================TOKEN===================
# class Token():
#     def __init__(self, STATE="final"):
#         self.STATE = STATE
#         self.mpc_questions = []
#         self.picture_questions = None
#         self.islr_questions = None
#         # self.image = None

#     def initialize_mpc_questions(self):
#         self.STATE = 'review'
#         review_questions = KC_MPC_QUESTIONS
#         list_length = len(review_questions)
#         mpc_idxs = np.random.choice(range(list_length), size=10, replace=False)
#         print('______________________________________________')
#         print('mpc_idxs: ',mpc_idxs, type(mpc_idxs))
#         print('______________________________________________')
#         for idx in mpc_idxs:
#             self.mpc_questions.append(review_questions[idx])


#     def initialize_open_questions(self):
#         self.STATE = 'open'
#         review_questions = OPEN_QUESTIONS
#         list_length = len(review_questions)
#         open_idxs = np.random.choice(range(list_length), size=4, replace=False)
#         # print('______________________________________________')
#         # print('mpc_idxs: ',mpc_idxs, type(mpc_idxs))
#         # print('______________________________________________')
#         for idx in open_idxs:
#             self.mpc_questions.append(review_questions[idx])

#     def initialize_image_questions(self):
#         self.STATE = 'chart'
#         review_questions = IMAGE_QUESTIONS
#         list_length = len(review_questions)
#         open_idxs = np.random.choice(range(list_length), size=2, replace=False)
#         # print('______________________________________________')
#         # print('mpc_idxs: ',mpc_idxs, type(mpc_idxs))
#         # print('______________________________________________')
#         for idx in open_idxs:
#             # print(review_questions[idx])
#             self.mpc_questions.append(review_questions[idx])

#====================TOKEN===================
# class Token():
#     def __init__(self, STATE="final"):
#         self.STATE = STATE
#         self.mpc_questions = []
#         self.open_questions = []
    
#     def initialize_mpc_questions(self, exam_type="final"):
#         if exam_type == "midterm1":
#             review_questions = MIDTERM1_MPC_QUESTIONS
#         elif exam_type == "midterm2":
#             review_questions = MIDTERM2_MPC_QUESTIONS
#         elif exam_type == "final":
#             # Combine both midterms for the final
#             review_questions = MIDTERM1_MPC_QUESTIONS + MIDTERM2_MPC_QUESTIONS
#         else:
#             raise ValueError("Invalid exam type selected.")

#         # Randomly select questions (e.g., 10 questions)
#         list_length = len(review_questions)
#         mpc_idxs = np.random.choice(range(list_length), size=min(10, list_length), replace=False)
#         self.mpc_questions = [review_questions[idx] for idx in mpc_idxs]

#     def initialize_open_questions(self, exam_type="final"):
#         if exam_type == "midterm1":
#             review_questions = MIDTERM1_OPEN_QUESTIONS
#         elif exam_type == "midterm2":
#             review_questions = MIDTERM2_OPEN_QUESTIONS
#         elif exam_type == "final":
#             # Combine both midterms for the final
#             review_questions = MIDTERM1_OPEN_QUESTIONS + MIDTERM2_OPEN_QUESTIONS
#         else:
#             raise ValueError("Invalid exam type selected.")
        
#         # Randomly select questions (e.g., 4 questions)
#         list_length = len(review_questions)
#         open_idxs = np.random.choice(range(list_length), size=min(4, list_length), replace=False)
#         self.open_questions = [review_questions[idx] for idx in open_idxs]

# import numpy as np

# class Token():
#     def __init__(self, STATE="final"):
#         self.STATE = STATE
#         self.mpc_questions = []
#         self.open_questions = []
#         self.image_questions = []

#     def initialize_mpc_questions(self, exam_type="final"):
#         if exam_type == "midterm1":
#             review_questions = MIDTERM1_MPC_QUESTIONS
#         elif exam_type == "midterm2":
#             review_questions = MIDTERM2_MPC_QUESTIONS
#         elif exam_type == "final":
#             review_questions = MIDTERM1_MPC_QUESTIONS + MIDTERM2_MPC_QUESTIONS
#         else:
#             raise ValueError("Invalid exam type selected.")

#         # Randomly select questions, up to a max of 10
#         list_length = len(review_questions)
#         mpc_idxs = np.random.choice(range(list_length), size=min(10, list_length), replace=False)
#         self.mpc_questions = [review_questions[idx] for idx in mpc_idxs]

#     def initialize_open_questions(self, exam_type="final"):
#         if exam_type == "midterm1":
#             review_questions = MIDTERM1_OPEN_QUESTIONS
#         elif exam_type == "midterm2":
#             review_questions = MIDTERM2_OPEN_QUESTIONS
#         elif exam_type == "final":
#             review_questions = MIDTERM1_OPEN_QUESTIONS + MIDTERM2_OPEN_QUESTIONS
#         else:
#             raise ValueError("Invalid exam type selected.")

#         # Randomly select open-ended questions, up to a max of 4
#         list_length = len(review_questions)
#         open_idxs = np.random.choice(range(list_length), size=min(4, list_length), replace=False)
#         self.open_questions = [review_questions[idx] for idx in open_idxs]

class Token():
    def __init__(self, STATE="final"):
        self.STATE = STATE
        self.mpc_questions = []
        self.open_questions = []
        self.image_questions = []

    def initialize_mpc_questions(self, exam_type="final"):
        # Initialize multiple-choice questions based on the selected exam type
        if exam_type == "midterm1":
            review_questions = MIDTERM1_MPC_QUESTIONS
        elif exam_type == "midterm2":
            review_questions = MIDTERM2_MPC_QUESTIONS
        elif exam_type == "final":
            review_questions = MIDTERM1_MPC_QUESTIONS + MIDTERM2_MPC_QUESTIONS
        else:
            raise ValueError("Invalid exam type selected.")

        # Select random questions, up to a max of 10
        list_length = len(review_questions)
        mpc_idxs = np.random.choice(range(list_length), size=min(10, list_length), replace=False)
        self.mpc_questions = [review_questions[idx] for idx in mpc_idxs]

    def initialize_open_questions(self, exam_type="final"):
        # Initialize open questions based on the selected exam type
        if exam_type == "midterm1":
            review_questions = MIDTERM1_OPEN_QUESTIONS
        elif exam_type == "midterm2":
            review_questions = MIDTERM2_OPEN_QUESTIONS
        elif exam_type == "final":
            review_questions = MIDTERM1_OPEN_QUESTIONS + MIDTERM2_OPEN_QUESTIONS
        else:
            raise ValueError("Invalid exam type selected.")

        # Select random questions, up to a max of 4
        list_length = len(review_questions)
        open_idxs = np.random.choice(range(list_length), size=min(4, list_length), replace=False)
        self.open_questions = [review_questions[idx] for idx in open_idxs]

    def initialize_image_questions(self):
        # Set state to 'chart' and load from IMAGE_QUESTIONS directly without exam type selection
        self.STATE = 'chart'
        review_questions = IMAGE_QUESTIONS

        # Select random questions, up to a max of 2
        list_length = len(review_questions)
        image_idxs = np.random.choice(range(list_length), size=min(2, list_length), replace=False)
        self.image_questions = [review_questions[idx] for idx in image_idxs]
