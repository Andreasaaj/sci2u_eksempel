from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch, Arc
import matplotlib.pyplot as plt
import numpy as np
from arrows3d import add_arrow3d

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]; x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]; y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]; z_mean = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])

# Ellipsoid
def ellipsoid(ax, rad_x=10, rad_y=10, rad_z=10, color='b'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = rad_x * np.outer(np.cos(u), np.sin(v))
    y = rad_y * np.outer(np.sin(u), np.sin(v))
    z = rad_z * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color)

# Cylinder
def cylinder(ax, rad=1, length=4, color='b', direction='z'):
    # Points up along the z-direction by default, but can be changed
    # to the 'x' or 'y' direction.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rad * np.outer(np.cos(u), np.ones(np.size(v)))
    y = rad * np.outer(np.sin(u), np.ones(np.size(v)))
    z = 0.5 * (length * np.outer(np.ones(np.size(u)), (v - np.pi / 2) / (np.pi / 2)) + length)

    # Top cover
    x_2 = rad * np.outer(np.cos(u), np.sin(v))
    y_2 = rad * np.outer(np.sin(u), np.sin(v))
    z_2 = 1.001 * length * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    if direction == 'z':
        ax.plot_surface(x, y, z, rstride=10, cstride=10, color=color, edgecolor='k', shade=False)
        ax.plot_surface(x_2, y_2, z_2, rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
        ax.plot_surface(x_2, y_2, -0.0001*z_2, rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
    elif direction == 'y':
        ax.plot_surface(y, z, x, rstride=10, cstride=10, color=color, edgecolor='k', shade=False)
        ax.plot_surface(y_2, z_2, x_2, rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
        ax.plot_surface(y_2, -0.0001*z_2, x_2, rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
    elif direction == 'x':
        ax.plot_surface(z, x, y,  rstride=10, cstride=10, color=color, edgecolor='k', shade=False)
        ax.plot_surface(z_2, x_2, y_2, rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
        ax.plot_surface(-0.0001*z_2, x_2, y_2,  rstride=10, cstride=50, color=color, edgecolor='k', shade=False)
# Box
def box(length=1, width=1, height=1, color='b'):
    # For drawing planes, just draw the bottom only.
    u = np.linspace(0, 1, 2)
    x, y = np.meshgrid(u, u)
    x *= length
    y *= width

    # Top
    ax.plot_surface(x, y, 0*x + height,  rstride=4, cstride=4, color=color)

    # Bottom
    ax.plot_surface(x, y, 0*x,  rstride=4, cstride=4, color=color)

    # Sides
    ax.plot_surface(x*0 + height, y, x,  rstride=4, cstride=4, color=color)
    ax.plot_surface(x, 0*y + height, y * height / width,  rstride=4, cstride=4, color=color)
    ax.plot_surface(0*x, y, x * height / length,  rstride=4, cstride=4, color=color)
    ax.plot_surface(x, 0*x, y * height / width,  rstride=4, cstride=4, color=color)

# Plane
def plane(length=1, width=2, color='b'):
    u = np.linspace(0, 1, 2)
    x, y = np.meshgrid(u, u)
    x *= length
    y *= width
    ax.plot_surface(x, y, 0*x,  rstride=4, cstride=4, color=color)

# Ellipse
def ellipse(rad_x=1, rad_y=1, height=0, color='b', direction='z'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rad_x * np.outer(np.cos(u), np.sin(v))
    y = rad_y * np.outer(np.sin(u), np.sin(v))
    z = height * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    if direction == 'z':
        ax.plot_surface(x, y, z, rstride=10, cstride=50, color=color, edgecolor=color, shade=False)
    elif direction == 'y':
        ax.plot_surface(y, z, x, rstride=10, cstride=50, color=color, edgecolor=color, shade=False)
    elif direction == 'x':
        ax.plot_surface(z, x, y,  rstride=10, cstride=50, color=color, edgecolor=color, shade=False)

def angle(angle_center=(0, 0, 0), starting_angle=0.0, ending_angle=75, angle_width=0.1, angle_height=0.1):
    p = Arc(angle_center[0:2], angle_width, angle_height, theta1=starting_angle, theta2=ending_angle)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=angle_center[-1], zdir="y")
#examples below:
#ellipsoid(10,20,10)
#cylinder(length=10, color='g', direction='y')
#box(5,10,5)
#ellipse(rad_x=2, direction='x')

## Angle example:
#add_arrow3d(ax, [0.5, 0.75], [1, 1], [0, 1])
#plane(color='r')
#angle((0.5, 0, 1), angle_width=0.5, angle_height=0.5)

#ax.set_aspect('equal')
#set_axes_equal(ax)
#ax.set_axis_off()
#plt.show()
