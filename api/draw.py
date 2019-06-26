# -*- coding: utf-8 -*-
# Python2 backports
from __future__ import absolute_import, division, generators, nested_scopes
from __future__ import print_function, unicode_literals, with_statement
# Numerics
import numpy as np
# Symbolics
import sympy as sp
from sympy.utilities.lambdify import lambdify
# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Other builtins
import sys
# The sci2u api
from api import plotting, latex


# Constants
sym = {'m' : sp.Symbol('m' , positive=True),
       's' : sp.Symbol('s' , positive=True),
       'kg': sp.Symbol('kg', positive=True),
       'J' : sp.Symbol('J' , positive=True),
       }
derived_units = {sym['kg']*sym['m']**2/sym['s']**2: sym['J']}


# A 2D grid (in 2D or 3D)
def grid(xmax,      xmin=None, xstep=None,
         ymax=None, ymin=None, ystep=None,
         step=1,
         ax=None,
         color='k',
         facecolor='k',
         facealpha=0,
         axes_order='xyz',
         ):
    if ax is None:
        ax = plt.gca()
    if ymax is None:
        ymax = xmax
    if xmin is None:
        xmin = -xmax
    if ymin is None:
        ymin = xmin
    if xstep is None:
        xstep = step
    if ystep is None:
        ystep = step
    if '3D' not in ax.__class__.__name__:
        # 2D plot.
        # Face filling.
        ax.fill((xmin, xmax, xmax, xmin), (ymin, ymin, ymax, ymax),
                facecolor=facecolor,
                alpha=facealpha,
                )
        # Lines
        for x in np.arange(xmin, xmax + xstep, xstep):
            ax.plot((x, x), (ymin, ymax), '-', color=color, linewidth=0.5)
        for y in np.arange(ymin, ymax + ystep, ystep):
            ax.plot((xmin, xmax), (y, y), '-', color=color, linewidth=0.5)
    else:
        # 3D plot.
        # Face filling.
        data = ((xmin, xmax, xmax, xmin),
                (ymin, ymin, ymax, ymax),
                (0   , 0   , 0   , 0   ))
        verts = [list(zip(data[axes_order.index('x')],
                          data[axes_order.index('y')],
                          data[axes_order.index('z')]))]
        poly = Poly3DCollection(verts, facecolors=[facecolor])
        poly.set_alpha(facealpha)
        ax.add_collection(poly)
        # Lines
        for x in np.arange(xmin, xmax + xstep, xstep):
            data = ((x, x), (ymin, ymax), (0, 0))
            ax.plot(data[axes_order.index('x')],
                    data[axes_order.index('y')],
                    data[axes_order.index('z')],
                    '-', color=color, linewidth=0.5,
                    clip_on=False,
                    )
        for y in np.arange(ymin, ymax + ystep, ystep):
            data = ((xmin, xmax), (y, y), (0, 0))
            ax.plot(data[axes_order.index('x')],
                    data[axes_order.index('y')],
                    data[axes_order.index('z')],
                    '-', color=color, linewidth=0.5,
                    clip_on=False,
                    )

# A labelled symbol of an axis going out of or into the paper
def third_axis(xlim, ylim, ax=None, label=r'$z$', direction='outwards'):
    if direction not in ('outwards', 'inwards'):
        print("The direction keyword must be either 'outwards' or 'inwards'.")
        sys.exit(1)
    if ax is None:
        ax = plt.gca()
    width  = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    symbol_pos = np.array((xlim[0] + 0.10*width,  ylim[0] + 0.08*height))
    label_pos = symbol_pos + np.array((-0.025*width, +0.09*height))
    ax.text(symbol_pos[0], symbol_pos[1],
            r'$\odot$' if direction == 'outwards' else r'$\otimes$',
            fontsize=plotting.TEXT_LARGE, horizontalalignment='left')
    ax.text(label_pos[0], label_pos[1],
            label,
            fontsize=plotting.TEXT_SMALL, horizontalalignment='left')

# Draw coordinate axes
def coordinate_axes(pos=(0, 0, 0),
                    size=0.25,
                    color='k',
                    include='xy',
                    labels=(r'$\hat{\imath}$', r'$\hat{\jmath}$', r'$\hat{k}$'),
                    axes_order='xyz',
                    ax=None,
                    ):
    if ax is None:
        ax = plt.gca()
    if not '3D' in ax.__class__.__name__:
        # 2D plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lengths = (xlim[1] - xlim[0],
                   ylim[1] - ylim[0],
                   )
        size *= np.mean(lengths)
        for axis in include:
            index = 'xy'.index(axis)
            label = labels[index]
            data = np.array(((pos[0], pos[0]),
                             (pos[1], pos[1]),
                             ),
                            dtype='double',
                            )
            # For some reason, the arrows are not plotted in their
            # entire length by default.
            data[index][0] -= 0.017*lengths[index]
            data[index][1] += 0.017*lengths[index]
            # Extrude arrow
            data[index][1] += size
            arrow_from = (data[0][0], data[1][0])
            arrow_to   = (data[0][1], data[1][1])
            ax.add_patch(patches.FancyArrowPatch(arrow_from, arrow_to, 
                                                 color=color,
                                                 linewidth=2, 
                                                 arrowstyle='-|>',
                                                 mutation_scale=18,
                                                 )
                        )
            # The axis label
            data = data.copy()
            data[index][1] += 0.14*lengths[index]
            ax.text(data[axes_order.index('x')][1],
                    data[axes_order.index('y')][1],
                    label,
                    fontsize=plotting.TEXT_SMALL,
                    horizontalalignment='center',
                    verticalalignment='center',
                    )

    else:
        # 3D plot
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        lengths = (xlim[1] - xlim[0],
                   ylim[1] - ylim[0],
                   zlim[1] - zlim[0],
                   )
        size *= np.mean(lengths)
        for axis in include:
            index = 'xyz'.index(axis)
            label = labels[index]
            data = np.array(((pos[0], pos[0]),
                             (pos[1], pos[1]),
                             (pos[2], pos[2]),
                             ),
                            dtype='double',
                            )
            # For some reason, the arrows are not plotted in their
            # entire length by default.
            data[index][0] -= 0.017*lengths[index]
            data[index][1] += 0.017*lengths[index]
            # Extrude arrow
            data[index][1] += size
            ax.add_artist(Arrow3D(data[axes_order.index('x')],
                                  data[axes_order.index('y')],
                                  data[axes_order.index('z')],
                                  color=color,
                                  linewidth=1.5, 
                                  arrowstyle='-|>',
                                  mutation_scale=18,
                                  clip_on=False,
                                  )
                          )
            # The axis label
            data = data.copy()
            data[index][1] += 0.14*lengths[index]
            ax.text(data[axes_order.index('x')][1],
                    data[axes_order.index('y')][1],
                    data[axes_order.index('z')][1],
                    label,
                    fontsize=plotting.TEXT_SMALL,
                    horizontalalignment='center',
                    verticalalignment='center',
                    )

# Arrows in 3D plots.
# This is needed because patches.FancyArrowPatch does not work in 3D.
class Arrow3D(patches.FancyArrowPatch):
    """To plot an Arrow3D, use ax.add_artist(Arrow3D(...))
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        patches.FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = ([float(el) for el in xs],
                         [float(el) for el in ys],
                         [float(el) for el in zs])
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        patches.FancyArrowPatch.draw(self, renderer)

# Equal axes in 3D
def axes_equal3D(ax=None):
    '''Equal scale axes via ax.set_aspect('equal') or ax.axis('equal')
    does not work in 3D. Call this function in stead.
    '''
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal')
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    x_range = abs(xlim[1] - xlim[0])
    x_middle = np.mean(xlim)
    y_range = abs(ylim[1] - ylim[0])
    y_middle = np.mean(ylim)
    z_range = abs(zlim[1] - zlim[0])
    z_middle = np.mean(zlim)
    r = 0.5*max((x_range, y_range, z_range))
    ax.set_xlim3d([x_middle - r, x_middle + r])
    ax.set_ylim3d([y_middle - r, y_middle + r])
    ax.set_zlim3d([z_middle - r, z_middle + r])

# For orthographic perspective
def orthogonal_proj(zfront, zback):
    """To set orthographic projection of current 3D axis, do
    proj3d.persp_transformation = draw.orthogonal_proj
    before doing any plotting.
    """
    a = (zfront + zback)/(zfront - zback)
    b = -2*zfront*zback/(zfront - zback)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, 0, zback]])

# A "3D" sphere
def sphere(X, Y, r, perspective=0.25, rot=0, label='', color='k', lw=2, ax=None):
    if ax is None:
        ax = plt.gca()
    N = 100
    # Circle
    q = np.linspace(0, 2*np.pi, N)
    x = X + r*np.cos(q)
    y = Y + r*np.sin(q)
    ax.plot(x, y, '-', color=color, lw=lw)
    # Flattened circle (behind)
    q = np.linspace(0, np.pi, N//2)
    x = r*np.cos(q)
    y = perspective*r*np.sin(q)
    rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    x, y = np.dot(rot_matrix, np.array([x, y]))
    x += X
    y += Y
    ax.plot(x, y, '--', color='k', lw=lw)
    # Flattened circle (in front)
    q = np.linspace(np.pi, 2*np.pi, N//2)
    x = r*np.cos(q)
    y = perspective*r*np.sin(q)
    rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    x, y = np.dot(rot_matrix, np.array([x, y]))
    x += X
    y += Y
    ax.plot(x, y, '-', color='k', lw=lw)
    # Radius label
    if label:
        q = 0.25*np.pi
        x = X + r*np.cos(q)
        y = Y + r*np.sin(q)
        ax.plot((X, x), (Y, y), '-', color=color, lw=lw)
        ax.text(np.mean((X, x)) - 0.4, np.mean((Y, y)) + 0.6,
                label,
                fontsize=plotting.TEXT_SMALL,
                horizontalalignment='center',
                verticalalignment='center',
                )

# A proper 3D sphere, or just part of one
def sphere3D(pos=None, theta=(0, np.pi), phi=(0, 2*np.pi), R=1,
             N=250, strides=10,
             color='b', edgecolor='k', linewidth=0.5, alpha=0.4,
             linestyle='-', wireframe_color='None',
             ax=None):
    if ax is None:
        ax = plt.gca()
    # Function for constructing Cartesian coordinates from limits of
    # spherical coordinates.
    def theta_phi_limits2xyz(theta, phi, N, pos):
        # Symbolic --> float
        theta = [float(el) for el in theta]
        phi   = [float(el) for el in phi  ]
        # 1D arrays of spherical coordinates of surface
        theta = np.linspace(theta[0], theta[1], N)
        phi   = np.linspace(phi  [0], phi  [1], N)
        # 2D-arrays of Cartesian coordiantes of surface
        x = R*np.outer(np.cos(phi), np.sin(theta))
        y = R*np.outer(np.sin(phi), np.sin(theta))
        z = R*np.outer(np.ones(N) , np.cos(theta))
        # Offset the coordinates. If no pos are given,
        # place the sphere (or spherical part) in the origin.
        if pos is None:
            pos = [-0.5*(np.max(x) + np.min(x)),
                   -0.5*(np.max(y) + np.min(y)),
                   -0.5*(np.max(z) + np.min(z)),
                  ]
        x = pos[0] + x
        y = pos[1] + y
        z = pos[2] + z
        # Stride numbers used for plotting
        stride_theta = int(np.round(np.pi/(max(theta) - min(theta))*N/strides))
        stride_phi   = int(np.round(np.pi/(max(phi  ) - min(phi  ))*N/strides))
        return x, y, z, stride_theta, stride_phi, pos
    # Call theta_phi_limits2xyz just to get pos
    x, y, z, stride_theta, stride_phi, pos = theta_phi_limits2xyz(theta, phi, N, pos)
    # Add optional wireframe of entire sphere
    if wireframe_color != 'None':
        x, y, z, stride_theta, stride_phi, _ = theta_phi_limits2xyz((0, np.pi), (0, 2*np.pi), N, pos)
        ax.plot_wireframe(x, y, z,
                          linestyle=linestyle,
                          rstride=stride_phi,
                          cstride=stride_theta,
                          color=wireframe_color,
                          lw=linewidth,
                          )
    # The spherical surface
    x, y, z, stride_theta, stride_phi, _ = theta_phi_limits2xyz(theta, phi, N, pos)
    ax.plot_surface(x, y, z,
                    rstride=stride_phi,
                    cstride=stride_theta,
                    color=color,
                    linestyle=linestyle,
                    edgecolor=edgecolor,
                    lw=linewidth,
                    alpha=alpha,
                    )


# A proper 3D cylinder, or just part of one
def cylinder3D(pos=None, phi=(0, 2*np.pi), R=1, L=1,
               N=250, strides=10, phi_rot=0,
               color='b', edgecolor='k', linewidth=0.5, alpha=0.4,
               linestyle='-', wireframe_color='None',
               ax=None):
    if ax is None:
        ax = plt.gca()
    # Function for constructing Cartesian coordinates from limits of
    # spherical coordinates.
    def phi_limits2xyz(phi, N, pos):
        # Symbolic --> float
        phi = [float(el) + phi_rot for el in phi  ]
        # 1D arrays of spherical coordinates of surface
        phi = np.linspace(phi[0], phi[1], N)
        # 2D-arrays of Cartesian coordiantes of surface
        x = R*np.outer(np.cos(phi), np.ones(N))
        y = R*np.outer(np.sin(phi), np.ones(N))
        z = L*np.outer(np.ones(N) , np.linspace(-0.5, +0.5, N))
        # Offset the coordinates. If no pos are given,
        # place the sphere (or spherical part) in the origin.
        if pos is None:
            pos = [-0.5*(np.max(x) + np.min(x)),
                   -0.5*(np.max(y) + np.min(y)),
                   -0.5*(np.max(z) + np.min(z)),
                  ]
        x = pos[0] + x
        y = pos[1] + y
        z = pos[2] + z
        # Stride numbers used for plotting
        stride_phi = int(np.round(2*np.pi/(max(phi) - min(phi))*N/strides))
        stride_z = int(2*np.round(N/strides))
        return x, y, z, stride_phi, stride_z, pos
    # Call phi_limits2xyz just to get pos
    x, y, z, stride_phi, stride_z, pos = phi_limits2xyz(phi, N, pos)
    # Add optional wireframe of entire cylinder
    if wireframe_color != 'None':
        x, y, z, stride_phi, stride_z, _ = phi_limits2xyz((0, 2*np.pi), N, pos)
        ax.plot_wireframe(x, y, z,
                          linestyle=linestyle,
                          rstride=stride_phi,
                          cstride=stride_z,
                          color=wireframe_color,
                          lw=linewidth,
                          )
    # The spherical surface
    x, y, z, stride_phi, stride_z, _ = phi_limits2xyz(phi, N, pos)
    ax.plot_surface(x, y, z,
                    rstride=stride_phi,
                    cstride=stride_z,
                    color=color,
                    linestyle=linestyle,
                    edgecolor=edgecolor,
                    lw=linewidth,
                    alpha=alpha,
                    )

# A proper 3D revolution body, or just part of one
def revolution_body3D(pos=None, phi=(0, 2*np.pi), R=1, L=1,
               N=250, strides=10, phi_rot=0,
               color='b', edgecolor='k', linewidth=0.5, alpha=0.4,
               linestyle='-', wireframe_color='None',
               var=None,
               func=None,
               x0=None,
               x1=None,
               ax=None):

    eval_func = np.vectorize(lambdify(var, func, modules=[str('numpy'),str('sympy')]))

    if ax is None:
        ax = plt.gca()
    # Function for constructing Cartesian coordinates from limits of
    # spherical coordinates.
    def phi_limits2xyz(phi, N, pos):
        # Symbolic --> float
        phi = [float(el) + phi_rot for el in phi  ]
        # 1D arrays of spherical coordinates of surface
        phi = np.linspace(phi[0], phi[1], N)
        # 2D-arrays of Cartesian coordiantes of surface
        xs = np.linspace(-0.5, +0.5, N)
        xs = np.linspace(x0, x1, N)
        x = L*np.outer(np.ones(N) , xs)
        y = R*np.outer(np.cos(phi), eval_func(xs))
        z = R*np.outer(np.sin(phi), eval_func(xs))
        # Offset the coordinates. If no pos are given,
        # place the sphere (or spherical part) in the origin.
        if pos is None:
            pos = [-0.5*(np.max(x) + np.min(x)),
                   -0.5*(np.max(y) + np.min(y)),
                   -0.5*(np.max(z) + np.min(z)),
                  ]
        x = pos[0] + x
        y = pos[1] + y
        z = pos[2] + z
        # Stride numbers used for plotting
        stride_phi = int(np.round(2*np.pi/(max(phi) - min(phi))*N/strides))
        stride_z = int(2*np.round(N/strides))
        return x, y, z, stride_phi, stride_z, pos
    # Call phi_limits2xyz just to get pos
    x, y, z, stride_phi, stride_z, pos = phi_limits2xyz(phi, N, pos)
    # Add optional wireframe of entire cylinder
    if wireframe_color != 'None':
        x, y, z, stride_phi, stride_z, _ = phi_limits2xyz((0, 2*np.pi), N, pos)
        ax.plot_wireframe(x, y, z,
                          linestyle=linestyle,
                          rstride=stride_phi,
                          cstride=stride_z,
                          color=wireframe_color,
                          lw=linewidth,
                          )
    # The spherical surface
    x, y, z, stride_phi, stride_z, _ = phi_limits2xyz(phi, N, pos)
    ax.plot_surface(x, y, z,
                    rstride=stride_phi,
                    cstride=stride_z,
                    color=color,
                    linestyle=linestyle,
                    edgecolor=edgecolor,
                    lw=linewidth,
                    alpha=alpha,
                    )

# For converting symbolic variable representing a physical quantity
# (like a value with a unit) into LaTeX math code. 
def physical_quantity_latex(value,
                            displaystyle=True,
                            display_unity_factor=False,
                            units=(),
                            substitute_derived_units=True,
                            ):
    value = sp.sympify(value)
    # Seperate numeric factor from variables and units
    factor_numeric = value.subs({symbol: 1 for symbol in value.free_symbols})
    factor_symbolic = value/factor_numeric
    # Use both numeric and symbolic factor?
    use_numeric = True
    use_symbolic = True
    if factor_symbolic == 1 or factor_symbolic == sp.nan:
        use_symbolic = False
    if factor_numeric == 1 and use_symbolic:
        # Keep redundant factor of 1?
        if not display_unity_factor:
            use_numeric = False
    elif factor_numeric == -1 and use_symbolic:
        # Move factor of -1 onto symbolic part?
        if not display_unity_factor:
            factor_symbolic *= -1
            use_numeric = False
    # Latexify numeric factor
    factor_numeric_latex = latex(factor_numeric) if use_numeric else ''
    # Latexify symbolic factor (use roman units)
    if use_symbolic:
        # Substitute units for derived units
        if substitute_derived_units:
            for bases, derived in derived_units.items():
                factor_symbolic_updated = factor_symbolic.subs(bases, derived)
                if factor_symbolic_updated != factor_symbolic:
                    factor_symbolic = factor_symbolic_updated
                    if str(derived) not in units:
                        units += (str(derived), )
        # Latexify
        factor_symbolic_latex = latex(factor_symbolic.subs({sp.Symbol(unit, positive=True): 'latex' + unit for unit in units}))
        factor_symbolic_latex_backup = factor_symbolic_latex
        interunit_space = r'\mkern+2mu'
        for unit in units:
            factor_symbolic_latex = factor_symbolic_latex.replace('latex' + unit, r'{}\mathrm{{{}}}'.format(interunit_space, unit))
        if factor_symbolic_latex.startswith(interunit_space):
            factor_symbolic_latex = factor_symbolic_latex[len(interunit_space):]
    else:
        factor_symbolic_latex = ''
    # Construct LaTeX string
    s = r'{displaystyle} {numeric} {space} {symbolic}'.format(displaystyle=(r'\displaystyle' if displaystyle else r'\textstyle'),
                                                            numeric=factor_numeric_latex,
                                                            space=('\,' if r'\mathrm' in factor_symbolic_latex else ''),
                                                            symbolic=factor_symbolic_latex,
                                                            )
    return s
