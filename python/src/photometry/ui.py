"""User Interface (UI) module.
"""


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from photutils import (RectangularAperture,
                       RectangularAnnulus)

import numpy as np

from src.photometry.hdu import HDUW


class TrailSelector:
    """Helps the user select a trail.
    """

    def __init__(self, height: int, semi_out: int) -> None:
        """Constructor.

        Args:
            height (int): _description_
            semi_out (int): _description_
        """
        self.clicks = 0
        self.height = height
        self.semi_out = semi_out

    def on_click(self, event):
        """Function to invoke upon clicking.

        Args:
            event (_type_): on-click event
        """
        if self.clicks == 1:
            self.event1 = event
        elif self.clicks == 2:
            self.event2 = event
            self._compute_properties()
        self.clicks +=1

    def _compute_properties(self):
        """Computes and populates properties.
        """
        self.centre =    [0.5*(self.event1.xdata + self.event2.xdata),
                          0.5*(self.event1.ydata + self.event2.ydata)]
        self.width = np.sqrt(np.power(self.event1.xdata - self.event2.xdata, 2) + np.power(self.event1.ydata - self.event2.ydata, 2))
        self.theta = np.arctan((self.event1.ydata - self.centre[1])/(self.event1.xdata - self.centre[0]))
        self.rectangular_aperture = RectangularAperture(positions=self.centre,
                                                        w=self.width,
                                                        h=self.height,
                                                        theta=self.theta)
        self.rectangular_annulus = RectangularAnnulus(  positions=self.centre,
                                                        w_in=self.width,
                                                        w_out=self.width + 2 * self.semi_out,
                                                        h_in=self.height,
                                                        h_out=self.height + 2 * self.semi_out,
                                                        theta=self.theta)

    def is_done(self) -> bool: 
        """Indicates whether the trail has been fully selected.

        Returns:
            bool: true if the trail has been fully selected and can be defined.
        """
        if self.clicks >= 3:
            return True
        return False
            
            
class UI:

    def __init__(self, hduw: HDUW) -> None:
        plt.ion()

        self.hduw = hduw        
        self.fig = plt.figure()
        self.ax = plt.subplot(projection=hduw.wcs)
        self.im = self.ax.imshow(   hduw.hdu.data,
                                    cmap='Greys',
                                    origin='lower',
                                    vmin=hduw.bkg_median-3*hduw.bkg_sigma,
                                    vmax=hduw.bkg_median+3*hduw.bkg_sigma)
        self.fig.colorbar(self.im, ax=self.ax)

    def add_sources(self, sources):
        self.ax.scatter(    sources['xcentroid'],
                            sources['ycentroid'],
                            color='yellow',
                            alpha=0.4)
        self.update()

    def select_trail(self, selector: TrailSelector):
        # Build with event handling: https://matplotlib.org/stable/users/explain/event_handling.html
        self.fig.canvas.mpl_connect('button_press_event', selector.on_click)
        while selector.is_done() == False:
            plt.waitforbuttonpress()
                                                
        self.ax.add_patch(selector.rectangular_aperture._to_patch(  fill=True,
                                                                    color='blue',
                                                                    alpha=0.5))

        self.ax.add_patch(selector.rectangular_annulus._to_patch(   fill=True,
                                                                    color='red',
                                                                    alpha=0.2))

        self.update()
        plt.waitforbuttonpress()

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        