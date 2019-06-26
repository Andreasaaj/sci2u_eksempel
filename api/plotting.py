from matplotlib import rc, rcParams
from helpers import latex


import matplotlib.colors as colors

# Convert to new matlab colors:
# https://se.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html
colors.colorConverter.colors['b'] = [0, 0.4470, 0.7410]
colors.colorConverter.colors['r'] = [0.6350, 0.0780, 0.1840]
colors.colorConverter.colors['g'] = [0.4660, 0.6740, 0.1880]
colors.colorConverter.colors['c'] = [0.3010, 0.7450, 0.9330]
colors.colorConverter.colors['m'] = [0.4940, 0.1840, 0.5560]
colors.colorConverter.colors['y'] = [0.9290, 0.6940, 0.1250]
colors.colorConverter.colors['o'] = [0.8500, 0.3250, 0.0980]



def pi2in(pixels):
    """
    Converts pixels to inches
    """
    return float(pixels) / DPI


def in2pi(inches):
    """
    Converts inches to pixels
    """
    return int(float(inches) * DPI)


DPI = 72.

HELP_WIDTH = 600
HELP_HEIGHT = 600
HELP_FIGSIZE = (
    pi2in(HELP_WIDTH),
    pi2in(HELP_HEIGHT)
)

QUESTION_WIDTH = 600
QUESTION_HEIGHT = 600
QUESTION_FIGSIZE = (
    pi2in(QUESTION_WIDTH),
    pi2in(QUESTION_HEIGHT)
)

ANSWER_WIDTH = 300
ANSWER_HEIGHT = 200
ANSWER_FIGSIZE = (
    pi2in(ANSWER_WIDTH),
    pi2in(ANSWER_HEIGHT)
)


# --------------------------------------------------------------------------- #
# -- MATPLOTLIB                                                            -- #
# --------------------------------------------------------------------------- #

TEXT_LARGE = 40
TEXT_MEDIUM = 35
TEXT_SMALL = 30
TEXT_TINY = 20


TEXT_ANSWER_LARGE = TEXT_LARGE
TEXT_ANSWER_MEDIUM = TEXT_MEDIUM
TEXT_ANSWER_SMALL = TEXT_SMALL
TEXT_ANSWER_TINY = TEXT_TINY

FONT_RATIO = 3.4/3.05

TEXT_QUESTION_LARGE = TEXT_LARGE * FONT_RATIO
TEXT_QUESTION_MEDIUM = TEXT_MEDIUM * FONT_RATIO
TEXT_QUESTION_SMALL = TEXT_SMALL * FONT_RATIO
TEXT_QUESTION_TINY = TEXT_TINY * FONT_RATIO



COLOR_BLACK = 'k'
COLOR_WHITE = 'w'
COLOR_GREEN = 'g'
COLOR_RED = 'r'
COLOR_CYAN = 'c'
COLOR_MAGENTA = 'm'
COLOR_YELLOW = 'y'
COLOR_BLUE = 'b'
COLOR_ORANGE = 'o'


def matplotlib_init():
    """
    Initiates matplotlib with commonl used
    """
    rc('font', **{'family':'sans-serif', 'sans-serif':['Arial']})
    rcParams.update({
        'ps.usedistiller' : 'xpdf',
        'axes.labelsize': TEXT_LARGE,
        'font.size': TEXT_LARGE,
        'legend.fontsize': TEXT_LARGE,
        'xtick.labelsize': TEXT_LARGE,
        'ytick.labelsize': TEXT_LARGE,
        'lines.linewidth': 0.5, #0.5points is minimum for aip
        'lines.markersize': 3,
        'legend.numpoints': 1,
        # Use the Computer Modern font
        'font.family' : 'serif',
        'font.serif'  : 'cmr10',
    })

def matplotlib_enable_latex():
    """
    Enables support for Latex strings in matplotlib annotations
    """
    rc('text', usetex=True)
    rcParams.update({
        'text.latex.unicode' : True,
        'text.latex.preamble' : [r'\usepackage{bm}',
                                 r'\usepackage{amssymb}',
                                 r'\usepackage{amsmath}',
                                 r'\usepackage{tabstackengine}',
                                 r'\usepackage{microtype}',
                                 r'\usepackage{xcolor}',
                                 r'\usepackage{xfrac}',
                                 r'\makeatletter',
                                 r'\let\@minipagerestore=\raggedright', #http://tex.stackexchange.com/questions/269308/how-do-i-make-text-raggedright-inside-all-miniboxes
				 r'\righthyphenmin = 2',
				 r'\hyphenpenalty = 0',
				 #r'\let\@minipagerestore=\sloppy',
                                 r'\makeatother',
                                 r'\usepackage{mathtools}'],})

def axes_in_plot(fig, ax, axis_labels_out=False, ticks=False, axis_color='k', axis_thickness=1,
                 x_label=r'$x$', y_label=r'$y$', custom_ticks=[], tick_width=1, tick_length=1,
                 ticks_without_labels=False, label_size=TEXT_SMALL, int_ticks=False,
                 square_grid=False, frame=False, x_tick_labels=[], y_tick_labels=[], max_x_tick_len=100,
                 max_y_tick_len=100, show_zero=False, ticks_length_fraction=1.0, internal_ticks=False,
                 grid_on = False, custom_grid={'x_list': [], 'y_list': [], 'x_lims': [], 'y_lims': [],
                'width': 0.25, 'color': 'k', 'linestyle': '-'}):
    """
    Given a figure and an axis instance, returns that axis instance with axis vectors added
    inside the plot.
    if axis_labels_out = True, then the axis labels will be in extension of the
    axis vectors.
    axis_thickness is the width of the axis vectors (default is 1).
    axis_color is the color of the axes and the ticks.
    x_label is the x-axis label (default is '$x$').
    y_label is the y-axis label (default is '$y$').
    label_size is the size of the labels. Default is plotting.TEXT_SMALL.
    if ticks = True, matplotlibs ticks are moved inside.
    custom_ticks can be supplied as a list of two lists, one for the x-axis,
        and one for the y-axis.
    tick_width is the width of the ticks (default is 1).
    tick_length is the length of the ticks (default is 1).
    if ticks_without labels = True, the ticks are drawn, but their labels are not.
    if square_grid = True, plot will be resized so a grid is a square grid. If
        this does not work satisfactorily, try ax.set_aspect('equal', 'datalim').
    if frame = False, ax object's frame will be removed (default is False).
    if show_zero = True ax will get a zero matching the other ticklabels.
    tick_length_fraction multiplies tick_length with given float. Needed for backwards compatibility.
    if internal_ticks = True, ticks will only be in first quadrant.
    if grid_on=True a normal grid will be added following the given ticks.
    custom grid requires grid_on. Can be specified by supplying a dictionary with the wanted options:
        lists of wanted grid lines and coordinates of grid corners and appearance.
        One has to supply x_list and y_list, the rest is optional

    """
    if max_x_tick_len < 1 or max_y_tick_len < 1:
        raise RuntimeError('maximum tick length must exceed 0')
    assert isinstance(ticks_length_fraction, float), 'tick_length_fraction must be float!'

    x_lims = list(ax.get_xlim())
    y_lims = list(ax.get_ylim())

    figW = fig.get_figwidth()
    figH = fig.get_figheight()
    position = ax.get_position(original=True)
    lowest, bottom, width, height = position.bounds
    if square_grid:
        if height*(x_lims[1]-x_lims[0]) >= width*(y_lims[1]-y_lims[0]):
            y_size = (x_lims[1]-x_lims[0])*figH*height/(figW*width)
            y_lims[0] = 0.5*(y_lims[0]+y_lims[1])-0.5*y_size
            y_lims[1] = 0.5*(y_lims[0]+y_lims[1])+0.5*y_size
            y_lims_changed = True
        else:
            x_size = (y_lims[1]-y_lims[0])*figW*width/(figH*height)
            x_lims = (0.5*sum(x_lims)-0.5*x_size,
                      0.5*sum(x_lims)+0.5*x_size)
            y_lims_changed = False
    x_axis_vec = [x_lims[0], 0, x_lims[1]-x_lims[0], 0]
    y_axis_vec = [0, y_lims[0], 0, y_lims[1]-y_lims[0]]
    X,Y,U,V = zip(*[x_axis_vec, y_axis_vec])
    ax.quiver(X,Y,U,V, angles='xy', scale_units='xy', scale=1,
              color=axis_color, width = axis_thickness*0.004)

    if axis_labels_out:
        x_label_pos = (1.05*x_lims[1],0)
        y_label_pos = (0, 1.05*y_lims[1])
        x_label_al = ('left', 'center')
        y_label_al = ('center', 'bottom')
    else:
        x_label_pos = (x_lims[1],0)
        y_label_pos = (0.01*(x_lims[1]-x_lims[0]), y_lims[1])
        x_label_al = ('right', 'bottom')
        y_label_al = ('left', 'top')

    ax.text(x=x_label_pos[0], y=x_label_pos[1], s=x_label,
            fontsize=label_size, color=axis_color,
            ha=x_label_al[0], va=x_label_al[1])
    ax.text(x=y_label_pos[0], y=y_label_pos[1], s=y_label,
            fontsize=label_size, color=axis_color,
            ha=y_label_al[0], va=y_label_al[1])
    
    if ticks and len(custom_ticks) == 0:
        if int_ticks:
            custom_ticks = [sorted(list(set([int(i) for i in ax.get_xticks()]))),
                            sorted(list(set([int(i) for i in ax.get_yticks()])))]
            print(custom_ticks)
        else:
            custom_ticks = [list(ax.get_xticks()), list(ax.get_yticks())]
    if len(custom_ticks) > 0:
        if square_grid:
            if y_lims_changed:
                custom_ydiff = abs(custom_ticks[1][1]-custom_ticks[1][0])
                while min(custom_ticks[1]) > y_lims[0] + custom_ydiff:
                    custom_ticks[1].append(min(custom_ticks[1])-custom_ydiff)
                while max(custom_ticks[1]) < y_lims[1] - custom_ydiff:
                    custom_ticks[1].append(max(custom_ticks[1])+custom_ydiff)
                while len(custom_ticks[0]) > max_y_tick_len:
                    custom_ticks[0] = custom_ticks[0][::2]
            else:
                custom_xdiff = custom_ticks[0][1]-custom_ticks[0][0]
                while min(custom_ticks[0]) > x_lims[0] + custom_xdiff:
                    custom_ticks[0].append(min(custom_ticks[0])-custom_xdiff)
                    #print(custom_ticks)
                while max(custom_ticks[0]) < x_lims[1] - custom_xdiff:
                    custom_ticks[0].append(max(custom_ticks[0])+custom_xdiff)
                while len(custom_ticks[0]) > max_x_tick_len:
                    custom_ticks[0] = custom_ticks[0][::2]
        half_xtick_len = 0.002*width*figW*(y_lims[1]-y_lims[0])*tick_length
        half_ytick_len = 0.002*height*figH*(x_lims[1]-x_lims[0])*tick_length

        # Added an internal ticks and tick_length tweak option - Martin Knudsen August 2018
        if not internal_ticks:
            for tick_num, xtick in enumerate(custom_ticks[0]):
                if xtick == 0:
                    continue
                ax.plot([xtick, xtick], [-half_xtick_len*ticks_length_fraction, half_xtick_len*ticks_length_fraction],
                    color=axis_color, lw=tick_width)
                if not ticks_without_labels and len(x_tick_labels) == 0:
                    ax.text(x=xtick, y=-half_xtick_len, s=r'$%s$' % latex(xtick),
                        fontsize=label_size,
                        color=axis_color,
                        ha='center', va='top')
                if len(x_tick_labels) > 0:
                    ax.text(x=xtick, y=-half_xtick_len, s=x_tick_labels[tick_num],
                        fontsize=label_size,
                        color=axis_color,
                        ha='center', va='top')

            for tick_num, ytick in enumerate(custom_ticks[1]):
                if ytick == 0:
                    continue
                ax.plot([-half_ytick_len*ticks_length_fraction, half_ytick_len*ticks_length_fraction], [ytick, ytick],
                    color=axis_color, lw=tick_width)
                if not ticks_without_labels and len(y_tick_labels) == 0:
                    ax.text(x=-half_ytick_len, y=ytick, s=r'$%s$' % latex(ytick),
                        fontsize=label_size,
                        color=axis_color,
                        ha='right', va='center')
                if len(y_tick_labels) > 0:
                    ax.text(x=-half_ytick_len, y=ytick, s=y_tick_labels[tick_num],
                        fontsize=label_size,
                        color=axis_color,
                        ha='right', va='center')

        elif internal_ticks:

            for tick_num, xtick in enumerate(custom_ticks[0]):
                if xtick == 0:
                    continue
                ax.plot([xtick, xtick], [-half_xtick_len*ticks_length_fraction, half_xtick_len*ticks_length_fraction],
                    color=axis_color, lw=tick_width)
                if not ticks_without_labels and len(x_tick_labels) == 0:
                    ax.text(x=xtick, y=-half_xtick_len, s=r'$%s$' % latex(xtick),
                        fontsize=label_size,
                        color=axis_color,
                        ha='center', va='top')
                if len(x_tick_labels) > 0:
                    ax.text(x=xtick, y=-half_xtick_len, s=x_tick_labels[tick_num],
                        fontsize=label_size,
                        color=axis_color,
                        ha='center', va='top')

            for tick_num, ytick in enumerate(custom_ticks[1]):
                if ytick == 0:
                    continue
                ax.plot([0., half_ytick_len*ticks_length_fraction], [ytick, ytick],
                            color=axis_color, lw=tick_width)
                if not ticks_without_labels and len(y_tick_labels) == 0:
                    ax.text(x=-half_ytick_len, y=ytick, s=r'$%s$' % latex(ytick),
                        fontsize=label_size,
                        color=axis_color,
                        ha='right', va='center')
                if len(y_tick_labels) > 0:
                    ax.text(x=-half_ytick_len, y=ytick, s=y_tick_labels[tick_num],
                        fontsize=label_size,
                        color=axis_color,
                        ha='right', va='center')
        ax.set_xticks(custom_ticks[0])
        ax.set_yticks(custom_ticks[1])
        
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    if not frame:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # Added a show_zero option which is closer to the axis than the other tick labels for visual serenity
    # - Martin Knudsen August 2018
    if show_zero:
        ax.text(x=-half_ytick_len*0.6, y=-half_xtick_len*0.6, s='$0$',
                fontsize=label_size,
                color=axis_color,
                ha='right', va='top')

    # Added grid - Martin Knudsen August 2018
    if grid_on:
        if len(custom_grid['x_list'])>0 or len(custom_grid['y_list'])>0:
            assert (len(custom_grid['y_list']) > 0 and len(
                custom_grid['x_list']) > 0), 'grid line lists cannot be empty'
            # default values of arguments. Only used if not supplied.
            # Limits default is an extension in each direction of two last gridlines.
            default_args = {'x_lims':[custom_grid['x_list'][0] - (custom_grid['x_list'][1]-custom_grid['x_list'][0]),
                                      custom_grid['x_list'][-1] + (custom_grid['x_list'][-1]-custom_grid['x_list'][-2])],
                            'y_lims':[custom_grid['y_list'][0] - (custom_grid['y_list'][1]-custom_grid['y_list'][0]),
                                      custom_grid['y_list'][-1] + (custom_grid['y_list'][-1]-custom_grid['y_list'][-2])],
                            'width': 0.25, 'color': 'k', 'linestyle': '-'}

            for arg in default_args.keys():
                if arg not in custom_grid:
                    custom_grid[arg]=default_args[arg]

            assert (len(custom_grid['x_lims'])==2
                    and len(custom_grid['y_lims'])==2), 'corners have to be in the form [x,y]'

            # to make sure no lines are outside limits
            custom_grid['x_list'] = [x for x in custom_grid['x_list']
                                     if (x >= custom_grid['x_lims'][0] and x <= custom_grid['x_lims'][1])]
            custom_grid['y_list'] = [x for x in custom_grid['y_list']
                                     if (x >= custom_grid['y_lims'][0] and x <= custom_grid['y_lims'][1])]

            ax.hlines(y=custom_grid['y_list'],
                  xmin=custom_grid['x_lims'][0],
                  xmax=custom_grid['x_lims'][1],
                  colors=custom_grid['color'],
                  linewidths=custom_grid['width']*axis_thickness,
                  linestyles=custom_grid['linestyle'],
                  zorder = -10.) # per default z-order for grid in background.
            ax.vlines(x=custom_grid['x_list'],
                  ymin=custom_grid['y_lims'][0],
                  ymax=custom_grid['y_lims'][1],
                  colors=custom_grid['color'],
                  linewidths=custom_grid['width']*axis_thickness,
                  linestyles=custom_grid['linestyle'],
                  zorder = -10.)
        else:
            ax.grid(color='k', linestyle='-', linewidth=axis_thickness*0.25, zorder=-10.)

    return None
