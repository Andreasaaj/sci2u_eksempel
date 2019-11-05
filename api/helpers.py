from sympy import Matrix, latex as sympy_latex
import sympy as sp
from re import findall

def latex(expr, big=False, **kwargs):
    """
    Returns a customized latex version of a sympy expression.
    """
    if type(expr)==type(Matrix()):
        raise TypeError("Matrices MUST be given to latex_matrix()")

    return_string = sympy_latex(expr, **kwargs)

    if big:
        return_string = r'\displaystyle %s' % return_string

    return return_string.replace(r'\log', r'\ln')

def latex_matrix(expr, parenthesis='(', equation_system=False, equation_dist=2, plus_dist = 5, minus_dist = 2, accomodate_fractions=True):
    """
    Returns a customized latex version of a sympy matrix.

    Takes a sympy matrix, parenthesis=<'(', '[', '|' or even '{'>,
        equation_system=<True|False> for augmented matrices,
        equation_dist=<an integer from 0 inf> is the distance
    between the last column before the line of the augmented matrix,
        plus_dist=<an integer from 0 inf> is the distance between
    row n and row n+1 if all entries in row n+1 are positive, and
        minus_dist=<an integer from 0 inf> is the distance between
    row n and row n+1 if some entries in row n+1 are negative.

    Default is parenthesis='(', equation_system=False, equation_dist=2, plus_dist = 3, minus_dist = 2

    Example: latex_matrix(Matrix([[0,1],[2,3],[4,5]]), plus_dist=2, minus_dist=1)
    yields a matrix with slightly less horizontal span than the default.
    """
    paren_dict = {'[': 'bracket', '(': 'paren', '|': 'vert', '{': 'brace'}
    if parenthesis in ['[', '(', '|', '{']:
        return_string = r'\setstacktabbedgap{15px}\fixTABwidth{T}' \
                        r'\%sMatrixstack[r]{' % paren_dict[parenthesis]
    else:
        raise RuntimeError("Parenthesis must be either '[', '(', '|' or '{'")

    for row in range(expr.rows):
        for col in range(expr.cols):
            if row == col == 0:
                return_string += latex(expr[row,col])
            elif col == 0:
                return_string += r'\\ %s' % latex(expr[row,col])
            elif equation_system and col == expr.rows - 1:
                return_string += r'& %s \big\vert' % latex(expr[row,col])
            else:
                return_string += r'& %s' % latex(expr[row,col])

    if accomodate_fractions and 'frac' in return_string:
        fractions_in_expr = tuple(findall(r'.*?\\\\', return_string))
        for frac_line in fractions_in_expr:
            if equation_system:
                return_string = return_string.replace(frac_line, frac_line+r'& '*(expr.cols-2)+r'\big\vert & \\')
            else:
                return_string = return_string.replace(frac_line, frac_line+r'\\')

    return_string += r'}'

    return return_string


def sympy_matrix2latex(m):
    latex_matrix = r'\arraycolsep=15pt'
    latex_matrix += sp.latex(m)
    latex_matrix = latex_matrix.replace(r'begin{matrix}', r'begin{array}{' + 'c' * m.cols + '} ')\
                                .replace(r'end{matrix}', r'end{array}')\
                                .replace(r'left[', r'left(')\
                                .replace(r'right]', r'right)')
    return latex_matrix

"""
from itertools import combinations, permutations
from sympy.parsing.sympy_parser import parse_expr


def get_simplest_expression(expression):
    pass


def combine(symbols, ops, length):
    unique_expr = []

    for sym in combinations(symbols, length):
        for op in permutations(ops, length - 1):
            res = []
            for i in xrange(length):
                res.append(str(sym[i]))
                if i < (length - 1):
                    res.append(str(op[i]))
            expr = parse_expr(''.join(res))
            if not expr in unique_expr:
                unique_expr.append(expr)
                yield expr

class ExpressionCombiner(object):
    def __init__(self, ops):
        self._combi = []
        self._ops = set(ops)

    def __iter__(self):
        return self

    def next(self):
        pass

    def add_symbols(self, expr, count):
        self._combi.append(combinations(expr, count))

    def get_combinations(self):
        for expr_combi in self._combi:
            return combine(expr_combi, self._ops, 4)


l1 = [a, b, c]
l2 = [cos(a), sin(a), -cos(a), -sin(a)]
ops = [+, -, *, /]


if __name__ == '__main__':
    from sympy import sin, cos, exp, diff, latex, simplify, symbols
    from sympy.abc import a, b, c, d

    sym = [a, b, c, d, cos(a), sin(a), -cos(a), -sin(a)]
    ops = symbols('*, +, -, /')

    for x in combine(sym, ops, 3):
        print(x)
"""
