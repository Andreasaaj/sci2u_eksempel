import random


def _repr_obj(obj):
    """
    Make a text representation of an object, like:
    
    ClassName(attr=value, attr=value, ...)
    """
    attrs = []
    for attr in dir(obj):
        val = getattr(obj, attr)
        if not attr.startswith('_') and not callable(val):
            attrs.append('%s=%s' % (attr, str(val)))
    s ='%s(%s)' % (obj.__class__.__name__, ', '.join(attrs))
    return s.replace('\n', ' ')


class QuestionBase(object):
    _default_group = 'Default'
    
    def __str__(self):
        return self.as_string()

    def __hash__(self):
        return hash(id(self)//16)
    
    def __eq__(self, other):
        return self.is_same_as(other)
    
    def __lt__(self, other):
        """
        Less difficult questions are lesser than others.
        Same difficulty is random ordered.
        """
        self_diff = self.difficulty()
        other_diff = other.difficulty()
        
        if self_diff == other_diff:
            return random.choice([True, False])
        else:
            return self.difficulty() < other.difficulty()
    
    def as_string(self):
        """
        Returns string representation of question.
        """
        return _repr_obj(self)
    
    def is_same_as(self, other_question):
        """
        Returns True if this question is equal to other_question,
        False otherwise.
        """
        raise NotImplementedError('Question must implement is_same_as()')
    
    def get_class(self):
        """
        Returns question group name.
        """
        return QuestionBase._default_group
    
    def sort_answers_randomly(self):
        """
        Returns True if answers to this question are allowed to be 
        sorted randomly, False otherwise.
        """
        return True
    
    def num_demanded_answers(self):
        """
        Returns the required number of correct answers to be for each question
        to be marked as correct. 0 means that all correct answers must be found.
        """
        return 1
    
    def num_answers(self):
        """
        Returns the number of answers to choose from for each question.
        """
        return 8
    
    def num_answers_nomalized(self):
        """
        Internal method - do not overwrite!
        Returns the number of answers to choose from.
        """
        if type(self.num_answers()) == dict:
            num_answers = 0
            for answer_type_str,num_answers_of_this_type in self.num_answers().items():
                num_answers += num_answers_of_this_type
        else:
            num_answers = self.num_answers()
        if num_answers < 2:
            raise Exception('num_answers() must return minimum 2')
        return num_answers
    
    def difficulty(self):
        """
        Evaluates question difficulty. Is used whenever questions
        are sorted by their difficulty.
        
        The higher return value, the more difficult the question.
        """
        return 0
    
    def proximate(self, answer):
        """
        Approximate how close an answer is to the question asked.
        Return value must be a float n, where 0 <= n <= 1.
        """
        return 0
    
    def max_similarity(self):
        """
        Returns the maximum allowed similarity between answers to this question.
        Return value must be a float n, where 0 <= n <= 1.
        """
        return 1

    def maximum_correct_answers(self):
        """
        Returns the maximum number of correct answers for this question.
        """
        return self.num_answers_nomalized()

    def correct_answer_proximate(self, answer):
        """
        Approximate how close a correct answer is to the question asked.
        Return value must be a float n, where 0 <= n <= 1.
        """
        return 0
    
    def make_answers(self):
        """
        Creates and returns answers to this question.
        Questions may generate any number of answers they like.
        Answers generated by this method does not need to be correct answers.
        
        make_answers() may return either an Answer-object (objects
        inherited from AnswerBase) or an iterator of Answer-objects.
        """
        raise NotImplementedError('Question must implement make_answers()')
    
    def draw(self, to_file):
        """
        Draw question. to_file is path of file to write image data to.
        """
        raise NotImplementedError('Question must implement draw()')
    
    def most_favorable_answer(self, answer1, answer2):
        """
        Decide whether answer1 or answer2 is most favorable
        as an answer to this question.
        
        Returns -1 if answer1 is more favorable than answer2.
        Returns 0 if answer1 and answer2 are equally favorable.
        Returns 1 if answer1 is less favorable than answer2.
        """
        raise NotImplementedError
    
    def favorized_answers(self, all_answers):
        """
        Given a large set of answers, this method returns a sorted list
        of answers, where the first elements (answers) are the most favored.
        Do not edit the original list 'all_answers' - in stead create a new!
        """
        raise NotImplementedError
    
    def pick_answers(self, all_answers):
        """
        Picks answers for this question.
        """
        raise NotImplementedError
    
    def validate_answers(self, picked_answers):
        """
        Check whether question has enough correct answers.
        Raises an exception if it doesn't.
        """
        raise NotImplementedError
    
    def answer_type(self,answer):
        """
        A string describing the answer type in the context of this question.
        """
        return ''

    # -------------------------------------------------------------------------
    
    # Legacy
    
    def get_group(self, *args, **kwargs):
        return self.get_class(*args, **kwargs)
    
    def required_correct_answers(self, *args, **kwargs):
        return self.num_demanded_answers(*args, **kwargs)


class AnswerBase(object):
    def __str__(self):
        return self.as_string()

    def __hash__(self):
        return hash(id(self)//16)

    def __eq__(self, other):
        return self.is_same_as(other)
    
    def __lt__(self, other):
        return self.is_less_than(other)
    
    def as_string(self):
        """
        Returns string representation of answer.
        """
        return _repr_obj(self)
    
    def is_same_as(self, other_answer):
        """
        Returns True if this answer is equal to other_answer,
        False otherwise.
        """
        raise NotImplementedError('Answer must implement is_same_as()')
    
    def similarity(self, other_answer):
        """
        Approximate how similar this answer is to another answer.
        Return value must be a float n, where 0 <= n <= 1.
        """
        return 0
    
    def is_less_than(self, other_answer):
        """
        Determine if this answer is lesser than other_answer.
        Used to sort out which order answers should appear in.
        """
        raise NotImplementedError('Answer must implement is_less_than()')
    
    def draw(self, to_file):
        """
        Draw answer. to_file is path of file to write image data to.
        """
        raise NotImplementedError('Answer must implement draw()')

class HelpBase(object):
    def draw(self, to_file):
        """
        Draw help. to_file is path of file to write image data to.
        """
        raise NotImplementedError('Help must implement draw()')
    
