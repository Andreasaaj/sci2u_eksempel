import functools
import random
from xml.dom import minidom

from question import QuestionBase, AnswerBase, HelpBase

import itertools, sys
from math import sqrt


# Method decorator to cache comparisons between two instances
def cache(func):
    """Applying this to the is_same_as method of Answer (and perhaps Question) 
    classes can sometimes greatly reduce computation time.
    """

    @functools.wraps(func)
    def wrapper(self, other):
        # If not already defined, instantiate cache as a class variable
        if not hasattr(self.__class__, 'cache'):
            setattr(self.__class__, 'cache', {})
        # Check cache for result
        cache = self.cache
        str_self = str(self)
        str_other = str(other)
        key = '{} == {}'.format(str_self, str_other)
        value = cache.get(key)
        if value is None:
            # Call function to get result
            value = func(self, other)
            # Update cache
            cache[key] = value
            key = '{} == {}'.format(str_other, str_self)
            cache[key] = value
        return value

    return wrapper


# Function for comparing two objects with arbitrary nested data
def compare_nested(a, b, recursion_lvl=0):
    # Exit on too deep a recursion
    if recursion_lvl > 100:
        print('Error in compare_nested')
        sys.exit(1)
    # Direct comparison
    try:
        comparison = (a == b)
        return comparison
    except:
        pass
    # Is a and b of the same length?
    try:
        len_a = len(a)
    except:
        len_a = 0
    try:
        len_b = len(b)
    except:
        len_b = 0
    if len_a != len_b:
        return False
    # Compare each element of a to each element of b
    for el_a, el_b in zip(a, b):
        comparison = compare_nested(el_a, el_b, recursion_lvl=(recursion_lvl + 1))
        if not isinstance(comparison, bool):
            try:
                comparison_tot = all(comparison)
            except:
                comparison_tot = comparison.all()
            comparison = comparison_tot
        if not comparison:
            return False
    return True


# Function which returns a generator over problem configurations,
# yielding N_questions selected evenly throughout the set of configurations.
def question_loop(configurations, N_questions):
    """configurations must be an iterable over all possible problem configurations.
    N_questions is the number of questions to include. This must be smaller or equal to the
    number of elements in the iteratable.
    """
    # Consume generator
    configurations = tuple(configurations)
    # Remove duplicates
    configurations_unique = []
    for configuration in configurations:
        duplicate = False
        for configuration_unique in configurations_unique:
            if compare_nested(configuration, configuration_unique):
                duplicate = True
                break
        if not duplicate:
            configurations_unique.append(configuration)
    configurations = configurations_unique
    # Check for too large N_questions
    size = len(configurations)
    if N_questions == 'all':
        N_questions = size
    if N_questions > size:
        print('{} questions requested but only {} defined!'.format(N_questions, size))
        sys.exit(1)
    # Determine the picking/skipping frequency
    skip_old = 1
    primes = (p for p in itertools.count(1) if not [t for t in range(2, 1 + int(sqrt(p))) if not p % t])
    for skip in primes:
        if size // skip <= N_questions:
            skip2 = skip
            skip = skip_old
            break
        skip_old = skip
    # Construct list of trimmed configurations
    indices = [i for i in range(0, N_questions * skip, skip)]
    indices2 = [i for i in range(0, N_questions * skip2, skip2) if i < size]
    for count, (i, i2) in enumerate(zip(reversed(indices), reversed(indices2)), 1):
        if i2 > i:
            indices[-count] = i2
        else:
            break
    configurations_trimmed = [configurations[i] for i in indices]
    return configurations_trimmed


class QuizQuestionBase(QuestionBase):
    def is_correct_answer(self, answer):
        """
        Returns True if answer is a correct answer to
        this question, False otherwise.
        """
        raise NotImplementedError('QuizQuestion must implement is_correct_answer()')

    def is_inappropriate_answer(self, answer):
        """
        Returns True if answer is an inappropriate answer to
        this question, False otherwise.
        """
        return False

    def most_favorable_answer(self, answer1, answer2):
        """
        Decide whether answer1 or answer2 is most favorable
        as an answer to this question.
        
        Returns -1 if answer1 is more favorable than answer2.
        Returns 0 if answer1 and answer2 are equal.
        Returns 1 if answer1 is less favorable than answer2.
        """
        if self.is_correct_answer(answer1) and self.is_correct_answer(answer2):
            apr1 = self.correct_answer_proximate(answer1)
            apr2 = self.correct_answer_proximate(answer2)
            if apr1 > apr2:
                return -1
            elif apr1 < apr2:
                return 1
            else:
                return random.choice([-1, 1])
        elif self.is_correct_answer(answer1):
            return -1
        elif self.is_correct_answer(answer2):
            return 1

        apr1 = self.proximate(answer1)
        apr2 = self.proximate(answer2)

        if apr1 > apr2:
            return -1
        elif apr1 < apr2:
            return 1
        else:
            return random.choice([-1, 1])

    def favorized_answers(self, all_answers):
        """
        Given a large set of answers, this method returns a sorted list
        of answers, where the first elements (answers) are the most favored.
        Do not edit the original list 'all_answers' - in stead create a new!
        """
        return sorted(all_answers, key=functools.cmp_to_key(self.most_favorable_answer))

    def pick_answers(self, all_answers):
        """
        Picks answers for this question.
        """
        picked_answers = []
        picked_answer_type_number_dict = {}
        num_answers = self.num_answers_nomalized()
        max_sim = self.max_similarity()
        max_ans = self.maximum_correct_answers()

        # Pick questions that are not too similar
        # Give a copy of the list to favorized_answers, so the
        # method can't mutate the all_answers.

        num_correct_ans = 0
        for answer in self.favorized_answers(list(all_answers)):
            is_too_similar = False
            is_not_too_correct = True

            #if not self.is_correct_answer(answer):
            if self.is_inappropriate_answer(answer):
                continue

            for a in picked_answers:
                if a.similarity(answer) > max_sim:
                    is_too_similar = True
                    break

            if self.is_correct_answer(answer):
                is_not_too_correct = num_correct_ans < max_ans
                if is_not_too_correct:
                    num_correct_ans += 1

            if not is_too_similar and is_not_too_correct:
                if not type(self.num_answers()) == dict or picked_answer_type_number_dict.get(self.answer_type(answer),0) < self.num_answers().get(self.answer_type(answer),0):
                    picked_answer_type_number_dict[self.answer_type(answer)] = picked_answer_type_number_dict.get(self.answer_type(answer),0) + 1
                    #print('Picked an answer for question {}. New type_dict: {}'.format(self,picked_answer_type_number_dict))
                    picked_answers.append(answer)

            if len(picked_answers) == num_answers:
                self.number_of_correct_answers = num_correct_ans
                break

        return sorted(picked_answers[:num_answers])

    def validate_answers(self, picked_answers):
        """
        Check whether question has enough correct answers.
        Raises an exception if it doesn't.
        """
        correct_required = self.required_correct_answers()
        correct_answers = [a for a in picked_answers if
                           self.is_correct_answer(a)]

        if len(correct_answers) < correct_required:
            raise RuntimeError('Question does not have enough correct answers: %s \n (%s found, %s required)' % (
                str(self),
                str(len(correct_answers)),
                str(correct_required)))


class QuizAnswerBase(AnswerBase):
    pass

class QuizHelpBase(HelpBase):
    pass

class QuizXMLReport(object):
    def __init__(self, sort_questions_randomly, title, help_sheets_ordered, help_sheets_dependent):
        self._doc = minidom.Document()
        self._root = self._doc.createElement('root')
        self._root.setAttribute('type', 'quiz')
        self._root.setAttribute('sort_questions_randomly',
                                '1' if sort_questions_randomly else '0')
        self._root.setAttribute('help_sheets_ordered',
                                '1' if help_sheets_ordered else '0')
        self._root.setAttribute('help_sheets_dependent',
                                '1' if help_sheets_dependent else '0')
        self._root.setAttribute('name', title)
        self._doc.appendChild(self._root)

    def add_metadata(self, keywords,description):
        meta = self._doc.createElement('meta')
        description_element = self._doc.createElement("description")
        description_element.setAttribute("text",description)
        meta.appendChild(description_element)
        for keyword in keywords:
            keyword_element = self._doc.createElement('keyword')
            keyword_element.setAttribute('text', keyword)
            meta.appendChild(keyword_element)
        self._root.appendChild(meta)

    def add_help(self, help_sheet):
        q = self._doc.createElement('question')
        q.setAttribute('filename', help_sheet._filename)
        q.setAttribute('required_correct_answers',
                       str(0))
        q.setAttribute('sort_answers_randomly',
                       '0' )
        self._root.appendChild(q)

    def add_question(self, question, answers):
        """
        Add a question and its answers to the report.
        """
        q = self._doc.createElement('question')
        q.setAttribute('filename', question._filename)
        req_corr_ans = question.required_correct_answers()
        if req_corr_ans == 0:
            req_corr_ans = question.number_of_correct_answers
        q.setAttribute('required_correct_answers',
                       str(req_corr_ans))
        q.setAttribute('sort_answers_randomly',
                       '1' if question.sort_answers_randomly() else '0')

        # Write answers
        N_corr_ans = 0
        for i, answer in enumerate(sorted(answers)):
            a = self._doc.createElement('answer')
            a.setAttribute('filename', answer._filename)
            a.setAttribute('sort', str(i))
            is_correct = question.is_correct_answer(answer)
            N_corr_ans += is_correct
            a.setAttribute('is_correct',
                           '1' if is_correct else '0')
            q.appendChild(a)
        # Check that the correct answers are flagged correctly
        if N_corr_ans != req_corr_ans and question.num_demanded_answers() == 0:
            print('\nError while adding question "{}" to xml report:\n'
                  '{} correct answers required but {} flagged as correct.'
                  .format(question, req_corr_ans, N_corr_ans)
                  )
            sys.exit(1)
        self._root.appendChild(q)

    def save(self, to_file):
        with open(to_file, 'w') as f:
            xml_str = self._doc.toprettyxml(indent='  ')
            f.write(xml_str)
