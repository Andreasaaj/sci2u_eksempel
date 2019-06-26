# -*- coding: utf-8 -*-
# Python2 backports
from __future__ import absolute_import, division, generators, nested_scopes
from __future__ import print_function, unicode_literals, with_statement
# Numerics
import numpy as np
# Symbolics
import sympy as sp
# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# The sci2u api
from api import latex, plotting
# Custom drawings
import draw



# Constants
sym = {'π' : sp.pi,
       'μ₀': sp.Symbol('mu0', positive=True),
       }

# Symbolic Euclidean norm
norm = lambda v: sp.sqrt(sum(v**2))

# Function for transforming arbitrary input(s) to a symbolic 3-vector
def symbolic_3vector(*v):
    v = np.array(np.ravel(v), dtype='object')
    # Enforce 3D
    while len(v) < 3:
        v = np.append(v, 0)
    # Enforce symbolic
    for i, el in enumerate(v):
        v[i] = sp.simplify(el)
    return v



# The point element class
class Point(object):
    # Instantiation
    def __init__(self, *pos):
        self.pos = symbolic_3vector(pos)

    # String representation
    def __str__(self):
        return '{}{}'.format(self.__class__.__name__,
                             self.pos,
                             ).replace('[', '(').replace(']', ')').replace(' ', ', ')
    # Hash from string representation
    def __hash__(self):
        return hash(str(self))

    # Arithmetic
    def __add__(self, other):
        other = self.__class__(other)
        return self.pos + other.pos
    def __sub__(self, other):
        if isinstance(other, InfiniteWire):
            return -(other - self)
        return self.pos - other
    def __mul__(self, other):
        return self.pos*other
    def __truediv__(self, other):
        return self.pos/other

    # Comparison
    def __eq__(self, other):
        other = self.__class__(other)
        return all(sp.simplify(a - b) == 0 for a, b in zip(self.pos, other.pos))

    # Container methods
    def __len__(self):
        return len(self.pos)
    def __getitem__(self, key):
        return self.pos[key]
    def __iter__(self):
        return self.pos.__iter__()

    # Physical quantities
    def norm(self):
        return norm(self.pos)

    # Plotting
    def draw(self, ax=None, subs={}, color='r', markersize=15, axes_order='xyz', **kwargs):
        if ax is None:
            ax = plt.gca()
        x, y, z = (float(sp.sympify(el).subs(subs)) for el in self.pos)
        if '3D' not in ax.__class__.__name__:
            # 2D plot
            ax.plot(x, y, '.', color=color, markersize=markersize, **kwargs)
        else:
            # 3D plot
            data = ([x], [y], [z])
            ax.plot(data[axes_order.index('x')],
                    data[axes_order.index('y')],
                    data[axes_order.index('z')],
                    '.', color=color, markersize=markersize,
                    clip_on=False,
                    **kwargs)
        

# The infinite wire element class
class InfiniteWire(object):
    # Instantiation
    def __init__(self, point, *current):
        # The current, its magnitude and its direction (a unit vector)
        self.current = symbolic_3vector(current)
        self.magnitude = norm(self.current)
        self.direction = self.current/self.magnitude
        # A point which the wire passes through.
        # If possible, shift the point along the wire onto the x-axis
        # or xy-plane. This removes the degeneracy between infinite
        # wires with different supplied points but corresponding to the
        # same infinite line.
        self.point = Point(point)
        for dim in range(3):
            if self.direction[dim]:
                t = self.point[dim]/self.direction[dim]
                self.point -= t*self.direction
        self.point = Point(self.point)

    # String representation
    def __str__(self):
        return '{}({}, current{})'.format(self.__class__.__name__,
                                          self.point,
                                          str(self.current).replace(' ', ', '),
                                          ).replace('[', '(').replace(']', ')')
    # Hash from string representation
    def __hash__(self):
        return hash(str(self))

    # Arithmetic
    def __sub__(self, point):
        point = Point(point)
        return (self.point - point) - np.dot(self.point - point, self.direction)*self.direction
    
    # Comparison
    def __eq__(self, other):
        return self.point == other.point and all(sp.simplify(a - b) == 0 for a, b in zip(self.current, other.current))
    def ontop(self, other):
        return self.point == other.point and (   all( self.direction == other.direction)
                                              or all(-self.direction == other.direction)
                                              )

    # Physical quantities
    def compute_Bfield(self, point):
        # B = μ₀/(2π|r|)I×rhat
        r = point - self
        B = sym['μ₀']/(2*sym['π']*norm(r))*np.cross(self.current, r/norm(r))
        return B

    # Plotting
    def draw(self, ax=None, subs={}, color='k', colorsegments=None,
             xboundary=(-10, +10), yboundary=(-10, +10), zboundary=(-10, +10),
             headplacement=0.8,
             axes_order='xyz',
             ring={},
             **kwargs):
        """The variables xboundary an yboundary defines a bounding
        rectangle for the "infinite" wires.
        The headplacement defines the placement of the arrowhead and
        should be between 0 and 1.
        Colorsegments is a sequence of (float, color) pairs,
        where the floats should be between 0 and 1. This is used to
        draw the wire in different colors.
        If supplied, ring should be a dict with the keys 'r', 'x', 'y'
        and 'z'. A ring with the specified radius will be placed at the
        specified position, rotated so that the wire penetrates it.
        This can be used to show wher the wire passes through a grid.
        """
        if ax is None:
            ax = plt.gca()
        if colorsegments is None:
            colorsegments = [(1, color)]
        plot3D = False
        if '3D' in ax.__class__.__name__:
            plot3D = True
        # Symbolic --> numeric
        point = np.array([float(sp.sympify(el).subs(subs)) for el in self.point.pos])
        direction = np.array([float(el) for el in self.direction])
        # If direction is exactly horizontal/vertical,
        # perturb it slightly (this avoids division by zero later).
        eps = np.finfo('double').eps
        for i, el in enumerate(direction):
            if abs(el) < 10*eps:
                direction[i] = 10*eps
        # Find "boundary points" of infinite wire
        boundary_points = [None, None]
        for dim, boundary in enumerate((xboundary, yboundary, zboundary)):
            if not plot3D and dim == 2:
                break
            tmin = (boundary[0] - point[dim])/direction[dim]
            tmax = (boundary[1] - point[dim])/direction[dim]
            candidate_start_point = point + tmin*direction
            candidate_end_point   = point + tmax*direction
            if (    abs(tmin/tmax) < 1e+6
                and xboundary[0] - 10*eps <= candidate_start_point[0] <= xboundary[1] + 10*eps
                and yboundary[0] - 10*eps <= candidate_start_point[1] <= yboundary[1] + 10*eps
                and zboundary[0] - 10*eps <= candidate_start_point[1] <= zboundary[1] + 10*eps
                ):
                # Accept start point
                boundary_points[0] = candidate_start_point
                Tmin = tmin
            if (    abs(tmax/tmin) < 1e+6
                and xboundary[0] - 10*eps <= candidate_end_point[0] <= xboundary[1] + 10*eps
                and yboundary[0] - 10*eps <= candidate_end_point[1] <= yboundary[1] + 10*eps
                and zboundary[0] - 10*eps <= candidate_end_point[1] <= zboundary[1] + 10*eps
                ):
                # Accept end point
                boundary_points[1] = candidate_end_point
                Tmax = tmax
        L = abs(Tmax - Tmin)
        # Plot line
        xend = boundary_points[0][0]
        yend = boundary_points[0][1]
        zend = boundary_points[0][2]
        arrowheadcolor_backward = None
        data = []
        for t, c in colorsegments:
            xstart = xend
            ystart = yend
            zstart = zend
            xend = boundary_points[0][0] + t*(boundary_points[1][0] - boundary_points[0][0])
            yend = boundary_points[0][1] + t*(boundary_points[1][1] - boundary_points[0][1])
            zend = boundary_points[0][2] + t*(boundary_points[1][2] - boundary_points[0][2])
            data.append(((xstart, xend),
                         (ystart, yend),
                         (zstart, zend),
                         )
                        )
            if headplacement <= t:
                arrowheadcolor_forward = c
            if headplacement >= t and arrowheadcolor_backward is None:
                arrowheadcolor_backward = c
        if not plot3D:
            # 2D
            for d, (t, c) in zip(data, colorsegments):
                ax.plot(d[0], d[1],
                        '-', color=c, linewidth=2, **kwargs)
        else:
            # 3D
            for d, (t, c) in zip(data, colorsegments):
                ax.plot(d[axes_order.index('x')],
                        d[axes_order.index('y')],
                        d[axes_order.index('z')],
                        '-', color=c, linewidth=2,
                        clip_on=False,
                        **kwargs)
            # Ring showing grid penetration.
            # See the accepted answer at http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            if ring:
                N = 50
                theta = np.linspace(0, 2*np.pi, N)
                x = ring['r']*np.cos(theta)
                y = ring['r']*np.sin(theta)
                z = np.zeros(N)
                a = np.array((0, 0, 1))
                b = direction
                v = np.cross(a, b)
                s = float(norm(v))
                c = np.dot(a, b)
                M = np.array((( 0   , -v[2],  v[1]),
                              ( v[2],  0   , -v[0]),
                              (-v[1],  v[0],  0),
                              )
                             )
                R = np.eye(3) + M + np.dot(M, M)/(1 + c + eps)
                data = np.dot(R, np.array((x, y, z)))
                data[0] += ring['x']
                data[1] += ring['y']
                data[2] += ring['z']
                ax.plot(data[axes_order.index('x')],
                        data[axes_order.index('y')],
                        data[axes_order.index('z')],
                        color=color,
                        linewidth=1,
                        )    
        # Make sure that the current flows from boundary_points[0] to boundary_points[1]
        arrowheadcolor = arrowheadcolor_forward
        if np.dot(boundary_points[1] - boundary_points[0], direction) < 0:
            boundary_points[0], boundary_points[1] = boundary_points[1], boundary_points[0]
            arrowheadcolor = arrowheadcolor_backward
        # Plot arrow head
        arrow_to   = boundary_points[0] + headplacement*L*direction
        arrow_from = arrow_to - 1e-6*(arrow_to - boundary_points[0])
        if not plot3D:
            ax.add_patch(patches.FancyArrowPatch(arrow_from[:2], arrow_to[:2], 
                                                 color=arrowheadcolor,
                                                 linewidth=2, 
                                                 arrowstyle='-|>',
                                                 mutation_scale=20,
                                                 clip_on=False,
                                                 )
                        )
        else:
            data = ((arrow_from[0], arrow_to[0]),
                    (arrow_from[1], arrow_to[1]),
                    (arrow_from[2], arrow_to[2]),
                    )
            ax.add_artist(draw.Arrow3D(data[axes_order.index('x')],
                                       data[axes_order.index('y')],
                                       data[axes_order.index('z')],
                                       color=arrowheadcolor,
                                       linewidth=2, 
                                       arrowstyle='-|>',
                                       mutation_scale=20,
                                       clip_on=False,
                                       )
                          )
        # Current label
        offset_fac_parallel = 0.024
        offset_fac_perpendicular = 0.024
        offset_parallel = (headplacement + offset_fac_parallel)*L*direction
        offset_perpendicular = offset_fac_perpendicular*L*np.array((direction[1], -direction[0], 0))
        if abs(direction[0]) < abs(direction[1]):
            # More of less vertical line.
            # Align to the right.
            horizontalalignment = 'right'
            if direction[1] < 0:
                # Direction downward. Shift label further down.
                offset_parallel[1] -= 0.05*L
            # Offset to the left
            fac = 2.0
            offset_perpendicular[0] = -fac*abs(offset_perpendicular[0])
        else:
            # More of less horizontal line
            if direction[0] < 0:
                # Direction to the left. Align to the right.
                horizontalalignment = 'right'
                #
                offset_parallel[0] -= 0.015*L
            else:
                # Direction to the right. Align to the left.
                horizontalalignment = 'left'
                #
                offset_parallel[0] += 0.015*L
            # Offset upwards
            offset_perpendicular[1] = +abs(offset_perpendicular[1])
        label_pos = boundary_points[0] + offset_parallel + offset_perpendicular
        if not plot3D:
            ax.text(label_pos[0], label_pos[1],
                    '${}$'.format(latex(self.magnitude)),
                    fontsize=plotting.TEXT_SMALL,
                    horizontalalignment=horizontalalignment,
                    )
        else:
            data = label_pos
            ax.text(data[axes_order.index('x')],
                    data[axes_order.index('y')],
                    data[axes_order.index('z')],
                    '${}$'.format(latex(self.magnitude)),
                    fontsize=plotting.TEXT_SMALL,
                    horizontalalignment=horizontalalignment,
                    )
