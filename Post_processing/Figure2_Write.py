import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
# Fixing random state for reproducibility
np.random.seed(19680801)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(1,0)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
n = 5

color_table=['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
count=0


def update_scatter(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([64])


def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(8)

for m, zlow, zhigh in [('o', 0, 1)]:

    #Group1 Stage2
    xs = [0.2880,0.6096,0.7038,0.6098,0.5214]#MCC
    ys = [1,0.9693,0.9323,0.8412,0.7921]#NPV
    zs = [1,0.9444,0.8444,0.5555,0.3555]#Sen

    #Group2 Stage1+2
    #xs = [1,0.9555,0.9222,0.9111,0.9111]#Sensitivy
    #ys = [1,0.9745,0.9617,0.9572,0.9572]#NPV
    #zs = [0.7654,0.8898,0.927,0.9387,0.9409]#AUC


    for i in range(5):
       distan=(1-xs[i])+(1-ys[i])+(1-zs[i])
       print(distan)
       #ax.scatter(xs[i], ys[i], zs[i], s=int(10 / (distan - 0.05)), marker=m, c=color_table[i], label='0.' + str(i + 1))
       ax.scatter(xs[i], ys[i], zs[i],s=int(10/(distan-0.45)), marker=m,c=color_table[i],label='0.'+str(i+1))

ax.set_xlabel('Matthews correlation coefficient')
ax.set_ylabel('Negative predictive value')
ax.set_zlabel('Sensitivity')
#ax.invert_xaxis()
#ax.invert_yaxis()
ax.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter),plt.Line2D: HandlerLine2D(update_func=updateline)},loc="upper left",markerscale=2., scatterpoints=1,title="Threshold")

plt.show()
#plt.savefig("")