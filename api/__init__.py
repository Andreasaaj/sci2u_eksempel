# For Python3 support
import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)

# API imports
import plotting
from core import generator
from quiz import QuizQuestionBase, QuizAnswerBase, QuizHelpBase, cache, compare_nested, question_loop
from puzzle import PuzzleQuestionBase, PuzzleAnswerBase, PuzzleTileBase
from helpers import latex, latex_matrix

# Plotting templates
sys.path.insert(0, this_dir + '/plotting_templates')
import misc_3d_shapes

# Initialize plotting
plotting.matplotlib_init()
plotting.matplotlib_enable_latex()

# Legacy support
QuestionBase = QuizQuestionBase
AnswerBase = QuizAnswerBase

