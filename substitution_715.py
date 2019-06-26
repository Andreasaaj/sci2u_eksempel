# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
from sympy import diff, Derivative, solve, Rational, fraction
from sympy import sin, cos, pi, exp, Integer, simplify, sqrt, Matrix, Transpose, Inverse, det, log
from sympy.abc import x,y,t,u
from sympy.utilities.lambdify import lambdify
import numpy as np
from api import plotting, QuestionBase, AnswerBase, generator
#from api import latex as l
from api import latex, latex_matrix
from random import shuffle
from sympy import integrate

"""
"""

DANISH = True
GROUP0 = '0'
GROUP1 = '1'
GROUP2 = '2'
GROUP3 = '3'
DIFF0 = 0
DIFF1 = 1
DIFF2 = 2
DIFF3 = 3

class Question(QuestionBase):

    def __init__(self, substitution, integrant, grp=0, dif=0):
        self.substitution = substitution
        self.integrant = integrant
        self.result = (integrate(integrant,u)).subs(u,substitution)
        self.grp = grp
        self.dif = dif

    def as_string(self):
        return 'substitution:  ' + str(self.substitution) + 'integrant:  ' + str(self.integrant) + ' result:' + str(self.result)

    def get_group(self):
        return self.grp

    def max_similarity(self):
        return 1.0

    def difficulty(self):
        return self.dif

    def make_answers(self):
        return Answer(self.result)

    #def required_correct_answers(self):
    #    return 2

    def num_answers(self):
        return 6

    def sort_answers_randomly(self):
        """
        Returns True if answers to this question are allowed to be 
        sorted randomly, False otherwise.
        """
        return True

    def proximate(self, answer):
        prox = 0
        return prox

    def is_same_as(self, other_question):
        return self.substitution == other_question.substitution and self.integrant == other_question.integrant

    def is_correct_answer(self, answer):
        return simplify(self.result - answer.value) == 0

    def draw(self, to_file):
        # Create figure
        if DANISH:
            text1 = r'Integralet $\displaystyle\int f(x)dx$, hvor $f$ er en funktion, ser efter '\
                r'substitutionen $\displaystyle {}={}$ således ud:'\
                r'$$\int {} du$$'\
                r'Hvad er $\displaystyle\int f(x)dx$?'.format(u,latex(self.substitution),latex(self.integrant))
        else:
            pass

        fig = plt.figure(figsize=plotting.QUESTION_FIGSIZE)


        text = r'\begin{{minipage}}{{550px}}' \
               r'{0} \\' \
               r'\end{{minipage}}'.format(text1)

        fig.text(x=.05,
                 y=.87,
                 s=text,
                 color=plotting.COLOR_BLACK,
                 horizontalalignment='left',
                 verticalalignment='top',
                 fontsize=plotting.TEXT_LARGE)

        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()


class Answer(AnswerBase):

    def __init__(self, result):
        self.value = result

    def as_string(self):
        return str(self.value)

    def is_same_as(self, other_answer):
        return simplify(self.value - other_answer.value) == 0

    def is_less_than(self, other_answer):
        return False

    def similarity(self, other_answer):
        # funktionens værdi i nul
        # eval_func = lambdify(self.q.var, self.q.func, modules=['numpy'])
        #self.q.func.subs(self.q.var,0)
        # dette svars værdi i nul
        # værdi sammenlignes med max_similarity, hvis større tages det ikke med
        return 0

    def draw(self, to_file):
        # Create figure
        if DANISH:
            text1 = r'$\displaystyle {0}$'.format(latex(self.value))
        else:
            pass

        fig = plt.figure(figsize=plotting.ANSWER_FIGSIZE)
        fig.text(x=.5,
                 y=.45,
                 s=text1,
                 color=plotting.COLOR_BLACK,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=plotting.TEXT_LARGE)

        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()
        
        # ----------------------------------------------------------------------------



def get_questions():
    mycounter = {}
    chosencounter = {}

    substitutions = [sqrt(x**2 + 1),
                     x**2 + 1,
                     1/(x**2 + 1),
                     ]
    integrants = [1/u,
                  sqrt(u),
                  ]
    difficulty = 0
    group = '0'
    trim = {difficulty: 1}
    for substitution in substitutions:
        for integrant in integrants:
            mycounter[difficulty] = mycounter.get(difficulty,0) + 1
            if mycounter[difficulty]%trim[difficulty] == 0:
                chosencounter[difficulty] = chosencounter.get(difficulty,0) + 1
                yield Question(substitution,integrant,group,difficulty)

#wrong_answers3 = [Answer(x*exp(x)),
#                  Answer(-x/exp(x)),
#]
#generator.extra_answers({GROUP0:wrong_answers0,GROUP3:wrong_answers3})
#generator.extra_answers({GROUP0:wrong_answers0,GROUP1:wrong_answers1})
#generator.enable_debugging()
generator.sort_questions_by_difficulty()
generator.ignore_duplicate_answers()
generator.ignore_duplicate_questions()
generator.generate(get_questions())
