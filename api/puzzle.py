import random
from xml.dom import minidom

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from question import QuestionBase, AnswerBase



class PuzzleQuestionBase(QuestionBase):
    def get_tiles(self):
        """
        Returns a list of PuzzleTile objects.
        """
        raise NotImplementedError('Question must implement get_tiles()')
    
    def draw_tiles(self, *args, **kwargs):
        for tile in self.make_tiles():
            tile.draw(*args, **kwargs)
    
    def most_favorable_answer(self, answer1, answer2):
        """
        Decide whether answer1 or answer2 is most favorable
        as an answer to this question.
        
        Returns -1 if answer1 is more favorable than answer2.
        Returns 0 if answer1 and answer2 are equal.
        Returns 1 if answer1 is less favorable than answer2.
        """
        apr1 = self.proximate(answer1) 
        apr2 = self.proximate(answer2)
        
        if apr1 > apr2:
            return -1
        elif apr1 < apr2:
            return 1
        else:
            return random.choice([-1, 1])
    
    def favorized_answers(self, all_answers):
        num_answers = self.num_answers_nomalized()
        max_sim = self.max_similarity()
        
        available_answers = [a for a in all_answers]
        correct_answers = []
        
        for tile in self._get_tiles():
            for answer in available_answers:
                if tile.is_correct_answer(answer):
                    correct_answers.append(answer)
                    available_answers.remove(answer)
        
        # Sort answers so that most favorable answers are first in the list
        return correct_answers + sorted(available_answers, 
                                        cmp=self.most_favorable_answer)

    def pick_answers(self, all_answers):
        """
        Picks answers for this question.
        """
        picked_answers = []
        
        # Pick questions that are not too similar
        for answer in self.favorized_answers(all_answers):
            is_too_similar = False
            
            for a in picked_answers:
                if a.similarity(answer) > max_sim:
                    is_too_similar = True
                    print('Too similar: %s AND %s' % (a.vector, answer.vector))
                    break
            
            if not is_too_similar:
                picked_answers.append(answer)
            
            if len(picked_answers) == num_answers:
                break
        
        return sorted(picked_answers[:num_answers])
    
    def validate_answers(self, picked_answers):
        """
        Check whether question has enough correct answers.
        Raises an exception if it doesn't.
        """
        for tile in self._get_tiles():
            has_correct_answer = False
            
            for answer in picked_answers:
                if tile.is_correct_answer(answer):
                    has_correct_answer = True
                    break
            
            if not has_correct_answer:
                raise RuntimeError('Tile does not have correct answer: %s' % str(tile))
    
    def _get_tiles(self):
        """
        Iterate over all tiles from a question
        """
        for tile in self.make_tiles():
            if not isinstance(tile, PuzzleTileBase):
                raise RuntimeError('Tile MUST inherit from PuzzleTileBase')
            yield tile


class PuzzleAnswerBase(AnswerBase):
    pass


class PuzzleTileBase(object):
    """
    Represents a single tile in a puzzle
    """
    def __str__(self):
        return self.as_string()
    
    def as_string(self):
        pass
    
    def is_correct_answer(self, answer):
        return False
    
    def center_x(self):
        pass
    
    def center_y(self):
        pass
    
    def width(self):
        pass
    
    def height(self):
        pass
    
    def scaling(self):
        return 1.0
    
    def draw(self, fig, color='w', alpha=0.5):
        center_x = float(self.center_x())
        center_y = float(self.center_y())
        width = float(self.width())
        height = float(self.height())
        
        btn_left = (
            center_x - (width / 2),
            center_y - (height / 2),
        )
        
        rect = Rectangle(
            btn_left,
            width,
            height,
            alpha=alpha, 
            color=color, 
            linestyle='dashed',
        )
        
        ax = fig.add_subplot(111, alpha=0.0)
        ax.axis('off')
        ax.add_patch(rect)


class PuzzleXMLReport(object):
    def __init__(self, sort_questions_randomly):
        self._doc = minidom.Document()
        self._root = self._doc.createElement('root')
        self._root.setAttribute('type', 'puzzle')
        self._root.setAttribute('sort_questions_randomly',
                                '1' if sort_questions_randomly else '0')
        self._doc.appendChild(self._root)
    
    def add_question(self, question, answers):
        """
        Add a question and its answers to the report.
        """
        q = self._doc.createElement('question')
        q.setAttribute('filename', question._filename)
        q.setAttribute('sort_answers_randomly', 
                    '1' if question.sort_answers_randomly() else '0')
        
        # Write answers
        ans = self._doc.createElement('answers')
        for i, answer in enumerate(sorted(answers)):
            a = self._doc.createElement('answer')
            a.setAttribute('filename', answer._filename)
            a.setAttribute('sort', str(i))
            ans.appendChild(a)
        
        # Write tiles
        til = self._doc.createElement('tiles')
        for tile in question.make_tiles():
            t = self._doc.createElement('tile')
            t.setAttribute('center_x', str(tile.center_x()))
            t.setAttribute('center_y', str(tile.center_y()))
            t.setAttribute('width', str(tile.width()))
            t.setAttribute('height', str(tile.height()))
            t.setAttribute('scaling', str(tile.scaling()))
            
            for answer in answers:
                if tile.is_correct_answer(answer):
                    t.setAttribute('answer', answer._filename)
                    break
            
            til.appendChild(t)
        
        q.appendChild(ans)
        q.appendChild(til)
        
        self._root.appendChild(q)
    
    def save(self, to_file):
        with open(to_file, 'w') as f:
            xml_str = self._doc.toprettyxml(indent='  ')
            f.write(xml_str)
