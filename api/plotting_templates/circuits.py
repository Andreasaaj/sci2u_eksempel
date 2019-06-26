import matplotlib.pyplot as plt
import SchemDraw as schem
import SchemDraw.elements as e


def circuit(ax):
    '''
    Draw a circuit with the SchemDraw package. It can be install with pip:
    'pip install SchemDraw'
    Full documentation and examples can be found here:
    'http://cdelker.bitbucket.org/SchemDraw/SchemDraw.html'
    '''
    d = schem.Drawing()
    V1 = d.add( e.SOURCE_V, label='5V' )
    d.add( e.LINE, d='right', l=d.unit*.75 )
    S1 = d.add( e.SWITCH_SPDT2_CLOSE, d='up', anchor='b', rgtlabel='$t=0$' )
    d.add( e.LINE, d='right', xy=S1.c,  l=d.unit*.75 )
    d.add( e.RES, d='down', label='$100\Omega$', botlabel=['+','$v_o$','-'] )
    d.add( e.LINE, to=V1.start )
    d.add( e.CAP, xy=S1.a, d='down', toy=V1.start, label='1$\mu$F' )
    d.add( e.DOT )
    d.draw(ax, showplot=False)


fig = plt.figure()
ax = fig.add_subplot(111)
circuit(ax)
plt.axis('off')
plt.show()
