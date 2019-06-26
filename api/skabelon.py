# -*- coding: cp1252 -*-
import matplotlib.pyplot as plt
from sympy import sin, cos, exp, diff, ln, sqrt, Integer
from sympy import simplify, div, symbol, Matrix
from sympy.abc import a, b, c, n, t, x
import random
import numpy as np
from re import findall

from api import plotting, generator, latex, latex_matrix
from api import QuizQuestionBase, QuizAnswerBase


class Question(QuizQuestionBase):
    def __init__(self, func, var):
        # Basis for at skabe et nyt element i klassen.
        self.func = func
        self.var = var
    
    def as_string(self):
        # Streng-representationen af spørgsmålet.
        return str(self.func)
    
    def get_group(self):
        # Undermappen, hvori spørgsmålet og svaret gemmes.
        return 'example_class'
    
    def make_answers(self):
        # Skaber svarerne.
        return Answer(self.func, self.var)
    
    def is_same_as(self, other_question):
        # Er dette dette spørgsmål det samme som other_question?
        return False
    
    def is_correct_answer(self, answer):
        """
        Is answer the correct answer? this method and is_same_as in Answer class
        are connected, so if one is changed, remember to change the other as
        well.
        """
        return True

    def max_similarity(self):
        """
        How closely may the answers resemble each other?
        """
        return 1.0

    def proximate(self, answer):
        """
        Determines the relevance of answers to the question.
        """
        return 0.0

    def num_answers(self):
        """Determines the number of answers for this question."""
        return 8

    def required_correct_answers(self):
        """
        Determines the number of correct answers to be found in order to
        move on from a question.
        """
        return 1

    def sort_answers_randomly(self):
        """
        Returns True if answers to this question are allowed to be 
        sorted randomly, False otherwise.
        """
        return False
    
    def draw(self, to_file):
        # Tegn spørgsmålet
        text = r'Example of equation: $\displaystyle %s $' % latex(self.func)
        
        fig = plt.figure(figsize=plotting.QUESTION_FIGSIZE)
        fig.text(x=.05,
                 y=.9,
                 s=r'Example text 1',
                 color=plotting.COLOR_BLACK,
                 fontsize=plotting.TEXT_LARGE)
        fig.text(x=.5,
                 y=.5,
                 s=text,
                 color=plotting.COLOR_BLACK,
                 fontsize=plotting.TEXT_LARGE,
                 horizontalalignment='center')
        fig.text(x=.05,
                 y=.25,
                 s=r'Example text 2',
                 color=plotting.COLOR_BLACK,
                 fontsize=plotting.TEXT_LARGE,
                 horizontalalignment='left',
                 verticalalignment='center')
        fig.text(x=.05,
                 y=.13,
                 s=r'Example text 3',
                 color=plotting.COLOR_BLACK,
                 fontsize=plotting.TEXT_LARGE,
                 horizontalalignment='left',
                 verticalalignment='center')
        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()


class Answer(QuizAnswerBase):
    def __init__(self, func, var):
        self.func = func
        self.var = var
    
    def as_string(self):
        return str(self.var)
    
    def is_same_as(self, other_answer):
        return False
    
    def is_less_than(self, other_answer):
        # Hvis svarene skal sorteres efter en bestemt logik, skrives den her.
        pass

    def similarity(self, other_answer):
        """
        How similar is this answer to other_answer? If returned value is larger
        than max_similarity in QuizQuestion, other_answer cannot be used for the
        same question.
        """
        return 0.0
    
    def draw(self, to_file):        
        text = latex(random.randrange(1,4)*self.var)
        if random.choice([0,1])==0:
            some_matrix = Matrix([[1,2,3],[4,Integer(3)/Integer(4),6],[-7,a,9]])
            text = latex_matrix(some_matrix)
            if random.choice([0,1])==0:
                text = latex_matrix(some_matrix, equation_system=True)
        
        fig = plt.figure(figsize=plotting.ANSWER_FIGSIZE)
        fig.text(x=.5,
                 y=.5,
                 s=r'$%s$' % text,
                 color=plotting.COLOR_BLACK,
                 fontsize=plotting.TEXT_LARGE,
                 horizontalalignment='center',
                 verticalalignment='center')
        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()


# ----------------------------------------------------------------------------


def get_functions(var):
    # Denne generator laver alle spørgsmålene.
    yield exp(a*var)
    yield ln(a*var)
    yield a*(var**4)
    yield a/(var**4)
    yield sqrt(1+(var**4))
    yield (1+var)**a
    yield exp(a*var**2)


def get_questions():
    # Sætter spørgsmålsgeneratoren i gang, og leverer resultatet til Question
    # klassen.
    for func in get_functions(x):
        yield Question(func, x)


generator.enable_debugging()
generator.ignore_duplicate_answers()
# argumentet i save_to() er den folder, du ønsker at gemme i (og er samtidig
# titlen på første linje i info.txt
generator.generate_quiz(get_questions())

