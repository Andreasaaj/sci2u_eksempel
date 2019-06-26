import plotting as pl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D, proj3d

"""
Template for creating the figures to electromagnetism.
both 2D- and 3D-plots can be chosen.

TODO:
    Add labels for arrows.
    Add basic shapes:
        Points, circles (inner/outer), spheres, planes, boxes.
    Make function to add a label with latex.
"""
class Plot_master():
    """
    Parent class for the plotting classes plot2D and plot3D.
    The figure sizes are pulled in from api/plotting.py, and the
    height is halved to use half the question size.
    The keyword 'square' makes the plot 300x300 pixels instead of
    the 600x300 to make square plots
    The keyword 'fullsize' makes the plot 600x600 pixels.
    """

    QUESTION_WIDTH = pl.QUESTION_WIDTH
    QUESTION_HEIGHT = pl.QUESTION_HEIGHT/2
    QUESTION_FIGSIZE = (pl.pi2in(QUESTION_WIDTH),
                        pl.pi2in(QUESTION_HEIGHT))

    def __init__(self, *args, **kwargs):
        if 'square' in args:
            self.QUESTION_FIGSIZE = (self.QUESTION_FIGSIZE[0]/2,
                                     self.QUESTION_FIGSIZE[1])
        if 'fullsize' in args:
            self.QUESTION_FIGSIZE = (self.QUESTION_FIGSIZE[0],
                                     self.QUESTION_FIGSIZE[1]*2)

        self.fig = plt.figure(figsize=self.QUESTION_FIGSIZE)

    def save_figure(self, file_path='./figure.png'):
        self.fig.savefig(file_path, dpi=pl.DPI)
        plt.close()

class Plot2D(Plot_master):

    def __init__(self, *args, **kwargs):
        Plot_master.__init__(self, *args, **kwargs)
        if 'small' in args:
            limits = (0.1, 0.1, 0.85, 0.85)
        else:
            limits = (0, 0, 1, 1)
        self.ax2D = self.fig.add_axes(limits)

#     def add_text(self, x=0, y=0, **kwargs)
# 
#         self.ax2D.text(x, 
#                        y, 
#                        string='X', 
#                        fontsize=20, 
#                        color='k',
#                        ha='center',
#                        va='center',
#                        **kwargs)

    def add_arrow2D(self,
                    x=[0, 0],
                    y=[0, 0],
                    headsize=20,
                    color='k',
                    lw=None):
        """
        Functions that adds a 2D-arrow to an ax-object.
        x- and y-coordinates are given as [start, end].
        If a line with no arrow-head is desired, you can set
        headsize=0.
        """

        if headsize == 0:
            self.ax2D.add_patch(FancyArrowPatch([x[0], y[0]],
                                                [x[1], y[1]],
                                                arrowstyle='-|>',
                                                lw=lw,
                                                color=color))
        else:
            self.ax2D.add_patch(FancyArrowPatch([x[0], y[0]],
                                                [x[1], y[1]],
                                                arrowstyle='-|>',
                                                mutation_scale=headsize,
                                                lw=lw,
                                                color=color))

    def add_circle(self, x=0.5, y=0.5, radius=0.1, color='r'):
        self.ax2D.add_patch(Circle((x,y), radius=radius, color=color))



class Plot3D(Plot_master):

    def __init__(self):
        Plot_master.__init__(self)
        self.ax3D = self.fig.add_axes([0, 0, 1, 1], projection='3d')

    def add_text(self, 
                 x=0, 
                 y=0, 
                 z=0, 
                 string=r'Text-string', 
                 **kwargs):

        self.ax3D.text(x, y, z,
                       string, 
                       color='k',
                       ha='center',
                       va='center',
                       **kwargs)

    def add_arrow3D(self, x=[0, 0], y=[0, 0], z=[0, 0], color='k'):
        """
        Function that adds a 3D-arrow to an ax-object.
        x-, y-, and z-coordinates are given with [start, end]
        """


        class Arrow3D(FancyArrowPatch):

            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(
                    self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(
                    xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)

        self.ax3D.add_artist(Arrow3D(x, y, z,
                                   mutation_scale=20,
                                   lw=1,
                                   arrowstyle='-|>',
                                   color=color))

# 
# pl2d = Plot2D('square')
# pl2d.add_arrow2D(x=[0.1, 0.9], y=[0.1, 0.1], lw=2)
# pl2d.add_arrow2D(x=[0.1, 0.9], y=[0.5, 0.5], lw=2)
# pl2d.add_arrow2D(x=[0.1, 0.9], y=[0.9, 0.9], lw=2)
# pl2d.add_circle()
# pl2d.add_text(x=0.7, y=0.7, string=r'$\otimes A$', fontsize=20, color='b')
# plt.axis('off')
# plt.show()
# pl2d.save_figure(file_path='../lol2d.png')

pl3d = Plot3D()
pl3d.add_arrow3D(x=[0.5, 1], y=[0.5, 1], z=[0.5, 1], color='g')
pl3d.add_arrow3D(x=[0.5, 0], y=[0.5, 0], z=[0.5, 0], color='r')
pl3d.add_arrow3D(x=[0.5, 0.7], y=[0.5, 0.7], z=[0.5, 0.1], color='b')
pl3d.add_text(x=0.5, y=0.5, z=0.5, string=r'$\otimes$', fontsize=30)
plt.show()
# pl3d.save_figure(file_path='../lol3d.png')
