import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from astropy.visualization import PercentileInterval, ImageNormalize
import numpy as np

from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import astropy.wcs.utils as utils
from astropy.visualization import PercentileInterval, ImageNormalize
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from astropy.nddata import Cutout2D



filename=get_pkg_data_filename('/Users/eracero/Documents/SolarSystem/aa_figures/Psyche/anonymous1552652728/hpacs_25HPPJSMAPB_blue_0429_p1830_00_v1.0_1470721482890.fits')

hdu = fits.open(filename)[1]
wcs = WCS(hdu.header)



fig = plt.figure(figsize=(8, 8), dpi=150)
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
fig.add_axes(ax)  

pp = 98.0
norm = ImageNormalize(hdu.data[~np.isnan(hdu.data)], interval=PercentileInterval(pp))
ax.imshow(hdu.data,norm=norm,cmap=plt.cm.gray,origin='lower',interpolation='nearest')
ax.grid(True)
#ax.set_xlabel('RA (J2000.0)',fontsize=12)
#ax.set_ylabel('Dec (J2000.0)',fontsize=12)

axins = zoomed_inset_axes(ax, 10, loc="lower right") # zoom = 6
#axins.imshow(ximage.data,norm=norm,cmap=plt.cm.viridis,origin='lower',interpolation='nearest')
# sub region of the original image
#x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
target_start = SkyCoord(ra=66.7658345, dec=18.948003, unit=(u.deg,u.deg), frame='icrs')
target_end= SkyCoord(ra=66.787029, dec=18.950426, unit=(u.deg,u.deg), frame='icrs')

#zoomSize = u.Quantity((10.0,10.0), u.arcmin)
#cutout = Cutout2D(ximage.data, target_coord, zoomSize, wcs=wcs)

pix_start = utils.skycoord_to_pixel(target_start, wcs=wcs)
pix_end = utils.skycoord_to_pixel(target_end, wcs=wcs)

axins.scatter(pix_start[0], pix_start[1], s=10,
            edgecolor='limegreen', linewidth=4, facecolor='limegreen')

axins.scatter(pix_end[0], pix_end[1], s=10,
            edgecolor='red', linewidth=4, facecolor='red')

#axins.imshow(cutout.data,norm=norm,cmap=plt.cm.gray,origin='lower',interpolation='nearest')
axins.imshow(hdu.data,norm=norm,cmap=plt.cm.gray,origin='lower',interpolation='nearest')
bottom, top = plt.ylim()
right, left = plt.xlim()
#print(bottom, top)

axins.set_xlim(left*0.63, left*0.69)
axins.set_ylim(top*0.68, top*0.74)

for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_linewidth(2)
    axins.spines[axis].set_color('white')
    
plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="white")

plt.savefig('psyche_pacs_blue.png',bbox_inches='tight')


plt.draw()
plt.show()