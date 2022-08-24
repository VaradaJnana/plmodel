import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors

"""
The code in this file was used once to generate an image of a colorbar, which is an asset that is used in the
website to aid in the visualization of certain numerical scores.
The code in this file is NOT called when the dashboard is run.
This code has simply been included here for reference. It is not crucial to the functioning of the dashboard,
and can be safely deleted.
"""

colorlist=["red", "darkorange", "yellow", "lawngreen", "green"]
newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=newcmp, norm=plt.Normalize(-1, 1))

plt.savefig('../templates/static/images/sentiment_score_colorbar.png', bbox_inches='tight')