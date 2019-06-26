import matplotlib.pyplot as plt
import matplotlib.patches as mp
from mpl_toolkits.mplot3d import Axes3D, proj3d

def add_arrow3d(ax, x=[0, 0], y=[0, 0], z=[0, 0], color='k'):
    """
    Function that adds a 3D-arrow to an ax-object.
    x-, y-, and z-coordinates are given with [start, end]
    """
    class Arrow3D(mp.FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            mp.FancyArrowPatch.__init__(self, 
                                       (0, 0), 
                                       (0, 0), 
                                       *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, 
                                               ys3d, 
                                               zs3d, 
                                               renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            mp.FancyArrowPatch.draw(self, renderer)

    ax.add_artist(Arrow3D(x, y, z,
                          mutation_scale=20,
                          lw=1,
                          arrowstyle='-|>',
                          color=color))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# add_arrow3d(ax, x=[0, 1], y=[0, 0], z=[0, 0])
# add_arrow3d(ax, x=[0, 0], y=[0, 1], z=[0, 0])
# add_arrow3d(ax, x=[0, 0], y=[0, 0], z=[0, 1])
# ax.text(1.05, 0, 0, '$x$', fontsize=20, ha='center', va='center')
# ax.text(0, 1.05, 0, '$y$', fontsize=20, ha='center', va='center')
# ax.text(0, 0, 1.05, '$z$', fontsize=20, ha='center', va='center')
# add_arrow3d(ax, x=[0.2, 0.7], y=[0, 0.8], z=[0.1, 0.7], color='b')
# plt.axis('off')
# plt.show()
