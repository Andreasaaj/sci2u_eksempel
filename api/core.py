#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
# foer jul
import os
import sys
import time
import shutil
import random
import hashlib

from html import make_html_preview
from quiz import QuizQuestionBase, QuizAnswerBase, QuizXMLReport
from puzzle import PuzzleQuestionBase, PuzzleAnswerBase, PuzzleXMLReport
#from importlib import reload

class FilenameList(list):
    i = 0

    def __init__(self, hashing=False):
        self._hashing = hashing

    def _md5_filename(self, s):
        name, ext = os.path.splitext(s)
        m = hashlib.md5()
        m.update(name.encode())
        m.update(str(time.time()).encode())
        m.update(str(random.randint(0, 10 ** 6)).encode())
        return '%s%s' % (m.hexdigest(), ext)

    def append(self, elem):
        elem._filename = self._make_filename(FilenameList.i)
        if self._hashing:
            elem._filename = self._md5_filename(elem._filename)

        list.append(self, elem)
        FilenameList.i += 1

    def _make_filename(self, i):
        raise NotImplementedError


class HelpList(FilenameList):
    def _make_filename(self, i):
        return 'h%04d.png' % i


class QuestionList(FilenameList):
    def _make_filename(self, i):
        return 'q%04d.png' % i


class AnswerList(FilenameList):
    def _make_filename(self, i):
        return 'a%04d.png' % i


class _Generator(object):
    ERROR = 0
    IGNORE = 1
    INCLUDE = 2
    RANDOM = 3
    DIFFICULTY = 4
    ORDERED = 5
    DEPENDENT = 6
    INDEPENDENT = 7

    def __init__(self):
        self._debugging = False
        self._duplicate_question = self.ERROR
        self._duplicate_answer = self.ERROR
        self._questions_sort = self.RANDOM
        self._help_sheets_ordered = self.ORDERED
        self._help_sheets_dependent = self.DEPENDENT
        self._question_type = None
        self._answer_type = None
        self.extra_answers_dictionary = {}

        # Added to support æ,ø,å and other utf-8 chars.
        #reload(sys)
        #sys.setdefaultencoding('utf8')

        # Determine which folder to place output files in.
        # Uses a subdirectory within the same directory, as the script
        # is placed. Directory is named the same as the script.
        folder, self._filename = os.path.split(sys.argv[0])
        filename, ext = os.path.splitext(self._filename)
        self._folder = os.path.join(os.getcwd(), folder, filename)

    def save_to(self, folder):
        self._folder = folder

    def enable_debugging(self):
        self._debugging = True

    # -- How to handle duplicate questions -----------------------------------

    def disallow_duplicate_questions(self):
        self._duplicate_question = self.ERROR

    def include_duplicate_questions(self):
        self._duplicate_question = self.INCLUDE

    def ignore_duplicate_questions(self):
        self._duplicate_question = self.IGNORE

    # -- How to handle duplicate answers -------------------------------------

    def disallow_duplicate_answers(self):
        self._duplicate_answer = self.ERROR

    def ignore_duplicate_answers(self):
        self._duplicate_answer = self.IGNORE

    def sort_questions_randomly(self):
        self._questions_sort = self.RANDOM

    def sort_questions_by_difficulty(self):
        self._questions_sort = self.DIFFICULTY

    def sort_help_sheets_randomly(self):
        self._help_sheets_ordered = self.RANDOM

    def help_sheets_are_independent(self):
        self._help_sheets_dependent = self.INDEPENDENT

    def help_sheets_release_one_by_one(self):
        self._help_sheets_dependent = self.INDEPENDENT

    # -- Processing ----------------------------------------------------------

    def generate_quiz(self, questions, help_sheets=[], keywords=[], title="",description=""):
        report = QuizXMLReport(self._questions_sort == self.RANDOM,
                               title,
                               self._help_sheets_ordered == self.ORDERED,
                               self._help_sheets_dependent == self.DEPENDENT)
        self._question_type = QuizQuestionBase
        self._answer_type = QuizAnswerBase
        self._process(questions, help_sheets, report, keywords,description)

    def generate_puzzle(self, questions):
        report = PuzzleXMLReport(self._questions_sort == self.RANDOM)
        self._question_type = PuzzleQuestionBase
        self._answer_type = PuzzleAnswerBase
        self._process(questions, report)

    def _process(self, q, help_sheets, report, keywords,description):
        print('----------------------------------------------------')
        print('sci2u.dk generator API')
        print('Mapping questions according to their group')

        helps = HelpList(hashing=not self._debugging)
        for i, help_sheet in enumerate(help_sheets):
            helps.append(help_sheet)
        all_helps = set()
        all_helps.update(helps)

        question_answer_pairs_final = []
        all_questions = set()
        all_answers = set()

        for group, questions_raw in self._group_questions(q):
            for extra_answer_group in self.extra_answers_dictionary.keys():
                if not extra_answer_group in self.all_groups:
                    raise RuntimeError('Extra answer group "%s" not a valid question group' % extra_answer_group)
            print('----------------------------------------------------')
            print('Processing %d questions in group "%s"' % (len(questions_raw), group))
            print('----------------------------------------------------')
            questions, answers = self._fetch(group, questions_raw)

            print('Group consists of %d questions and %d answers' % (len(questions), len(answers)))
            print('----------------------------------------------------')
            print('Picking answers to questions in group "%s"' % group)
            question_answer_pairs = []

            # Pick answers for each question in group
            for i, question in enumerate(questions):
                print('Picking answers for question %d of %d: %s' % (
                    i + 1, len(questions), str(question)))
                num_answers = question.num_answers_nomalized()
                picked_answers = question.pick_answers(answers)

                if len(picked_answers) < num_answers:
                    raise RuntimeError('Too few answers were picked for question: %s (got %d, expected %d)' % (
                        str(question), len(picked_answers), num_answers))
                elif len(picked_answers) > num_answers:
                    raise RuntimeError('Too many answers were picked for question: %s (got %d, expected %d)' % (
                        str(question), len(picked_answers), num_answers))

                # Make sure all picked answers are among all answers in group
                for answer in picked_answers:
                    if not answer in answers:
                        raise RuntimeError('Question (%s) picked unknown answer: %s' % (
                            str(question), str(answer)))

                question.validate_answers(picked_answers)
                question_answer_pairs.append((question, picked_answers))

            print('Group has %d questions which share %d answers' % (
                len(questions),
                len(answers),
            ))

            question_answer_pairs_final.extend(question_answer_pairs)
            all_questions.update(questions)
            all_answers.update(answers)

        # Sort question-answer-pairs by Question (difficulty).
        # Lowest difficulty is first in the list.
        print('----------------------------------------------------')
        print('Sorting all questions by difficulty')
        question_answer_pairs_final.sort(key=lambda qa: qa[0])

        print('----------------------------------------------------')

        # Delete directory and all of it's contents (if it exists)
        if os.path.isdir(self._folder):
            print('Truncating directory: %s' % self._folder)
            shutil.rmtree(self._folder)

        # Create directory
        os.makedirs(self._folder)

        # Include .py file in directory
        shutil.copy(self._filename, self._folder)

        # Draw questions and answers
        print('Drawing images to %s' % self._folder)
        self._draw(all_helps,'help')
        self._draw(all_questions, 'question')
        self._draw(all_answers, 'answer')

        # Write reports
        print('----------------------------------------------------')
        print('Writing combination files to %s' % self._folder)
        print('Writing export file: combinations.xml')

        # Add the keywords in a meta parent element in the top of the xml document and description.
        report.add_metadata(keywords,description)

        for help_sheet in helps:
            report.add_help(help_sheet)

        for question, answers in question_answer_pairs_final:
            report.add_question(question, answers)
            # Check that question and answer files exist
            question_file = os.path.join(self._folder, question._filename)
            if not os.path.isfile(question_file):
                print('\nError: Reference to question file "{}" which does not exist'.format(question_file))
                sys.exit(1)
            for answer in answers:
                answer_file = os.path.join(self._folder, answer._filename)
                if not os.path.isfile(answer_file):
                    print('\nError: Reference to answer file "{}" which does not exist'.format(answer_file))
                    sys.exit(1)


        report.save(os.path.join(self._folder, 'combinations.xml'))

        print('Writing preview file: combinations.html')
        # Sorts answers randomly for preview.
        question_answer_pairs_final_preview = list(question_answer_pairs_final)
        for question, answers in question_answer_pairs_final_preview:
            if question.sort_answers_randomly():
                random.shuffle(answers)

        make_html_preview(
            helps,
            question_answer_pairs_final_preview,
            os.path.join(self._folder, 'combinations.html'),
        )

        print('Done! Bye')
        print('----------------------------------------------------')

    def _group_questions(self, questions):
        """
        Group questions by their group.
        Returns [(group, [questions]), ...]
        """
        groups = {}

        for question in questions:
            if not isinstance(question, self._question_type):
                raise RuntimeError('Questions MUST inherit from %s' %
                                   self._question_type.__name__)

            group = question.get_group()

            if not isinstance(group, str):
                raise RuntimeError('Question group must be of type string (got %s)' %
                                   type(group).__name__)

            groups.setdefault(group, [])
            groups[group].append(question)

        self.all_groups = groups.keys()
        return groups.items()

    def _fetch(self, group, questions):
        """
        Fetch unique questions and answers from an iterator 
        of Question objects
        """
        questions_ignored = 0
        questions_duplicated = 0
        answers_ignored = 0

        questions_final = QuestionList(hashing=not self._debugging)
        answers_final = AnswerList(hashing=not self._debugging)

        for i, question in enumerate(questions):
            # Check whether question already exists within group and take
            # appropriate action if it does. Matching is done according
            # to Question's implementation of how two questions are equal
            if question in questions_final:
                if self._duplicate_question == self.IGNORE:
                    print('Ignoring duplicate question: %s' % str(question))
                    questions_ignored += 1
                    continue
                elif self._duplicate_question == self.INCLUDE:
                    print('Including duplicate question: %s' % str(question))
                    questions_duplicated += 1
                    pass
                else:
                    raise RuntimeError('Duplicate question in group "%s": %s' % (
                        group, str(question)))

            # At this point the question is accepted in the final pool
            questions_final.append(question)

            # Get answers from the question
            for answer in self._get_answers(question):
                # Check whether answer already exists and take appropriate
                # action if it does. Matching is done according to Answer's
                # implementation of how two answers are equal
                if answer in answers_final:
                    if self._duplicate_answer == self.IGNORE:
                        print('Ignoring duplicate answer: %s' % str(answer))
                        answers_ignored += 1
                        continue
                    else:
                        raise RuntimeError('Duplicate answer in group "%s": %s' % (
                            group, str(answer)))

                # At this point the answers is accepted in the final pool
                answers_final.append(answer)

            print('Processing %d%% complete (%d of %d)' % (
                (float(i + 1) / len(questions) * 100), i + 1, len(questions)))

        # Any question independent extra answers in group are included here.
        extra_answer_list = self.extra_answers_dictionary.get(group, [])
        if not len(extra_answer_list) == 0:
            print("\nIncluding %d extra answers:" % len(extra_answer_list))

        for answer in extra_answer_list:
            if not isinstance(answer, self._answer_type):
                raise RuntimeError("Extra answers for group '%s' MUST be of class Answer." % group)

            print('Answer:\t{}'.format(answer))
            if answer in answers_final:
                if self._duplicate_answer == self.IGNORE:
                    print('Ignoring duplicate answer: %s' % str(answer))
                    answers_ignored += 1
                    continue
                else:
                    raise RuntimeError('Duplicate answer in group "%s": %s' % (
                        group, str(answer)))

            # At this point the answers is accepted in the final pool
            answers_final.append(answer)

        if (self._duplicate_question == self.IGNORE
            or self._duplicate_answer == self.IGNORE):
            print('%d questions and %d answers were ignored because of duplicates' % (
                questions_ignored,
                answers_ignored,
            ))
        if self._duplicate_question == self.INCLUDE:
            print('%d duplicate questions were included' % questions_duplicated)

        return questions_final, answers_final

    def _get_answers(self, question):
        """
        Iterate over all answers from a question.
        Invokes make_answers() on question and validates the return type.
        """
        answers = question.make_answers()

        try:
            iter(answers)
        except TypeError:
            answers = [answers]

        for answer in answers:
            if not isinstance(answer, self._answer_type):
                raise RuntimeError('Answer MUST inherit from %s' % self._answer_type.__name__)

            yield answer

    def extra_answers(self, answer_dict):
        """
        Extra, non question specific answers.
        """

        for group, answer_list in answer_dict.items():
            if not isinstance(answer_list, list):
                raise TypeError("Dictionary values for extra answers MUST be lists of Answers.")
            if not isinstance(group, str):
                raise TypeError("Dictionary keys for extra answers MUST be groups of type 'string'.")

        self.extra_answers_dictionary = answer_dict

    def _draw(self, items, type_name):
        """
        Invoke draw on all items in the list 'items'.
        'type_name' is a string identifying what type 
            of item it is (question/answer).
        """
        for i, item in enumerate(items):
            print('Drawing %s %d of %d: %s => %s' % (
                type_name, i + 1, len(items), str(item), item._filename))
            filepath = os.path.join(self._folder, item._filename)
            item.draw(filepath)

            if not self._debugging and not os.path.isfile(filepath):
                raise RuntimeError('No image was drawn: %s' % filepath)

        print('Drawing of %d %ss complete' % (len(items), type_name))

    # Legacy
    disallow_dublicate_questions = disallow_duplicate_questions
    include_dublicate_questions = include_duplicate_questions
    ignore_dublicate_questions = ignore_duplicate_questions
    disallow_dublicate_answers = disallow_duplicate_answers
    ignore_dublicate_answers = ignore_duplicate_answers
    generate = generate_quiz


# Legacy
generator = _Generator()
