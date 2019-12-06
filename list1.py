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
    def __init__(self, group, mylist, result):
        self.group = group
        self.mylist = mylist
        self.result = result

    def as_string(self):
        return self.mylist

    def get_group(self):
        return self.group

    def max_similarity(self):
        return 1.0

    def difficulty(self):
        return 0

    def make_answers(self):
        return Answer(self.result)

    #def required_correct_answers(self):
    #    return 2

    def num_answers(self):
        return 8

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
        return self.mylist == other_question.mylist

    def is_correct_answer(self, answer):
        return self.result == answer.value

    def draw(self, to_file):
        # Create figure
                #                r'$$\mathtt{{{}}}$$' \



        if DANISH:
            text1 = r"For " \
                    r"\begin{{center}}\texttt{{a = {}}}\end{{center}} " \
                    r"Hvordan inkrementeres der med 1, for det sidste element i listen?".format(self.as_string())
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
                 fontsize=plotting.TEXT_MEDIUM)

        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()


class Answer(AnswerBase):

    def __init__(self, result):
        self.value = result

    def as_string(self):
        return str(self.value)

    def is_same_as(self, other_answer):
        return self.value == other_answer.value

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
            text1 = r'\texttt{{{}}}'.format(self.value)
        else:
            pass

        fig = plt.figure(figsize=plotting.ANSWER_FIGSIZE)
        fig.text(x=.5,
                 y=.45,
                 s=text1,
                 color=plotting.COLOR_BLACK,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=plotting.TEXT_SMALL)

        fig.savefig(to_file, dpi=plotting.DPI)
        plt.close()
        
        # ----------------------------------------------------------------------------






def get_questions():

    #lists = [[r"\lbrack 8\, 5\, 22 \rbrack "], [r"\lbrack 2\, 4\, 6 \rbrack "], [r"\lbrack 0\, 2\, 1 \rbrack "]]
    lists = [["[8, 5, 22]"], ["[2, 4, 6]"], ["[0, 2, 1]"]]
    answers_string = "a[2] += 1"

    for group_id, mylist in enumerate(lists):
        group = str(group_id)
        for type_ in lists[group_id]:
            yield Question(group, type_, answers_string)





wrong_answers0 = [Answer("a[2]+1"), Answer("a(2)+1"), Answer("a[3]+1"),
                  Answer("a(3)+1"), Answer("a[3] += 1"), Answer("a(2) = 1"),
                  Answer("a(3) = 1"), Answer("a[2] = 1")]

#generator.extra_answers({GROUP0:wrong_answers0})
generator.extra_answers({GROUP0:wrong_answers0, GROUP1:wrong_answers0, GROUP2:wrong_answers0})
#generator.enable_debugging()
generator.sort_questions_by_difficulty()
generator.ignore_duplicate_answers()
generator.ignore_duplicate_questions()
generator.generate(get_questions())
