import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Arc
import numpy as np



def misc2d(ax):
    """
    Function that uses matplotlib patches to draw common shapes
    The input is a matplotlib ax-object.

    Examples of shapes can be found here:
    http://matplotlib.org/examples/shapes_and_collections/artist_reference.html
    """
    # Simple text which can take latex-expressions and therefore be used to plot symbols.
    ax.text(0.3, 0.7, r'$\otimes A$', fontsize=20)
    # Circle shape which needs: center(x,y), radius.
    ax.add_patch(mp.Circle((0.5,0.5), radius=0.3, color='k', alpha=0.7, fill=False))
    ax.add_patch(mp.Circle((0.5,0.5), radius=0.2, color='r', alpha=0.7))
    ax.add_patch(mp.Circle((0.5,0.5), radius=0.1, color='w', alpha=None))
    # Arrow-shape. Here used to make x-y-axes. There are a lot of different settings, and they need
    # to be tweaked to fit in the frame as only half the arrows on the axes are shown.
    ax.add_patch(mp.FancyArrowPatch((0, 0), (1, 0), 
                                    color='k',
                                    lw=2, 
                                    arrowstyle='-|>',
                                    mutation_scale=20))
    ax.text(1, 0, '$x$', fontsize=20)
    ax.add_patch(mp.FancyArrowPatch((0, 0), (0, 1), 
                                    color='k',
                                    lw=2, 
                                    arrowstyle='-|>',
                                    mutation_scale=20))
    ax.text(0, 1, '$y$', fontsize=20)
    ax.add_patch(mp.FancyArrowPatch((0.5, 0.5), (0.1, 0.1), 
                                    color='b',
                                    lw=2, 
                                    arrowstyle='-|>',
                                    mutation_scale=20))
    # Simple rectangle
    ax.add_patch(mp.Rectangle((0.6, 0.6), 0.2, 0.2, alpha=0.5))

    # Simple polygon
    verts = [
    (0., 0.), # left, bottom
    (0., .25), # left, top
    (.25, 0.), # right, bottom
    (0., 0.), # ignored
    ]

    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='orange', alpha=0.5, lw=2)
    ax.add_patch(patch)

    # Plots an angle.
    starting_angle = 0.0
    ending_angle = 230
    angle_width, angle_height = 0.1, 0.1
    angle_center = (0.5, 0.5)
    p = Arc(angle_center, angle_width, angle_height, theta1=starting_angle, theta2=ending_angle)
    ax.add_patch(p)

    # Plots angle text.
    ax.text(0.5, 0.54, r'$\theta$', fontsize=20, ha='center', va='bottom')
    curved_line_with_arrow()
    curved_line_with_arrow((0.15, 0.3), radius=0.2, ending_angle=np.deg2rad(223), arrow_pos=0.8, arrow_direction=1)

def curved_line_with_arrow(center=(0.45, 0.25), radius=0.6, starting_angle=0.0, ending_angle=np.pi/3, arrow_pos=0.5,
                           arrow_direction=1):
    # Plots an angle.
    p = Arc(center, radius, radius, theta1=starting_angle, theta2=np.rad2deg(ending_angle))
    ax.add_patch(p)
    midway = 0.5*radius*np.array([np.cos(ending_angle*arrow_pos), np.sin(ending_angle*arrow_pos)])
    extra_angle=cmp(arrow_direction, 0)*np.pi/10
    midway_after = 0.5*radius*np.array([np.cos((ending_angle+extra_angle)*arrow_pos),
                                        np.sin((ending_angle+extra_angle)*arrow_pos)])
    midway += np.array([center[0], center[1]])
    midway_after += np.array([center[0], center[1]])
    ax.add_patch(mp.FancyArrowPatch(midway, midway_after,
                                    color='k',
                                    lw=2,
                                    arrowstyle='->',
                                    mutation_scale=20))

fig = plt.figure()
ax = fig.add_subplot(111)
misc2d(ax)
plt.axis('off')
plt.axis('equal')
plt.show()
