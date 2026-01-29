"""
Improved User Interface (UI) for trail selection with zoom/pan and live preview.
This version DOES NOT finalize on left click; only Enter finalizes.
"""
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
# photutils ≥ 1.0 moved aperture APIs under photutils.aperture
try:
    from photutils.aperture import RectangularAperture, RectangularAnnulus
except Exception:
    # very old photutils (<1.0)
    from photutils import RectangularAperture, RectangularAnnulus
    
from typing import Optional, Tuple, List
from astropy.io import fits

from src.photometry.hdu import HDUW

from astropy.coordinates import SkyCoord
import astropy.units as u
try:
    from astropy.wcs import WCS  # opcional, por tipado/claridad
except Exception:
    WCS = None

class TrailSelector:
    def __init__(self, height: float = 5.0, semi_out: float = 5.0,
                 finalize_on_click: bool = False) -> None:
        # User-adjustable geometry
        self.height = float(height)
        self.semi_out = float(semi_out)

        # Background placement
        self.annulus_centre_override = None  # optional (x,y)
        self.bg_box_centre = None             # (x,y) for external background
        self.use_external_bg = False

        # Behavior flag
        self.finalize_on_click = bool(finalize_on_click)

        # Internal state
        self._start = None
        self._end = None
        self.done = False
        self.canceled = False

        # Derived geometry
        self.centre = None
        self.width = None
        self.theta = None
        self.rectangular_aperture = None
        self.rectangular_annulus = None

    # ------------------------------------------------------------------

    def _compute_from_points(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = (x2 - x1), (y2 - y1)
        width = float(np.hypot(dx, dy))
        if width == 0:
            return

        self.centre = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
        self.width = width
        self.theta = np.arctan2(dy, dx)

        # Decide annulus centre
        if self.use_external_bg and self.bg_box_centre is not None:
            ann_centre = self.bg_box_centre
        elif self.annulus_centre_override is not None:
            ann_centre = self.annulus_centre_override
        else:
            ann_centre = self.centre

        self.rectangular_aperture = RectangularAperture(
            positions=self.centre,
            w=self.width,
            h=self.height,
            theta=self.theta
        )

        self.rectangular_annulus = RectangularAnnulus(
            positions=ann_centre,
            w_in=self.width,
            w_out=self.width + 2 * self.semi_out,
            h_in=self.height,
            h_out=self.height + 2 * self.semi_out,
            theta=self.theta
        )

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return bool(self.done)

    # ------------------------------------------------------------------

    def adjust_height(self, delta: float):
        self.height = max(1.0, self.height + float(delta))
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    def adjust_annulus(self, delta: float):
        self.semi_out = max(0.5, self.semi_out + float(delta))
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    # ------------------------------------------------------------------
    # Background control
    # ------------------------------------------------------------------

    def set_annulus_centre(self, x: float, y: float) -> None:
        """Recenter annulus without creating an external background box."""
        self.annulus_centre_override = (float(x), float(y))
        self.use_external_bg = False
        self.bg_box_centre = None
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    def start_background_box(self, x: float, y: float) -> None:
        """Place an external background box centred at (x, y)."""
        self.bg_box_centre = (float(x), float(y))
        self.use_external_bg = True
        self.annulus_centre_override = None
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    def reset_annulus_centre(self) -> None:
        """Reset background region to follow the trail aperture centre."""
        self.annulus_centre_override = None
        self.bg_box_centre = None
        self.use_external_bg = False
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def aperture(self):
        """Compatibility alias for photometry.py."""
        return self.rectangular_aperture

    @property
    def annulus(self):
        """Compatibility alias for photometry.py."""
        return self.rectangular_annulus

    def as_dict(self):
        return {
            "aperture": self.rectangular_aperture,
            "annulus": self.rectangular_annulus,
        }

    # ------------------------------------------------------------------

    def finalize(self):
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)
        self.done = True

    def set_wcs(self, wcs):
        self.wcs = wcs

class UI:
    def __init__(self, hduw: HDUW) -> None:
        # Preview patches (managed by UI)
        self._ap_patch = None
        self._an_patch = None

        self._hint_text = ""
        # UI state
        self._pan_press = None
        self._selector = None
        self._hint = None
        self._outside_mode = False   # NEW: outside-background mode
        
        # Allow second left click to freeze the plot and free the cursor
        self._selection_frozen = False

        self.hduw = hduw
        self.fig = plt.figure()
        self.ax = plt.subplot(projection=hduw.wcs)
        
        # Intenta obtener WCS desde el envoltorio o desde el HDU primario
        self.wcs = getattr(hduw, "wcs", None)
        if self.wcs is None and hasattr(hduw, "hdu") and hasattr(hduw.hdu, "header"):
            try:
                self.wcs = WCS(hduw.hdu.header) if WCS is not None else None
            except Exception:
                self.wcs = None     

        # assume self.image2d already exists; ensure it's masked
        img = np.ma.masked_invalid(hduw.hdu.data)
        self.im = self.ax.imshow(img, origin="lower", cmap="gray")
        self.im.cmap.set_bad(alpha=0)

        self.im = self.ax.imshow(
            hduw.hdu.data, cmap='Greys', origin='lower',
            vmin=hduw.bkg_median - 3 * hduw.bkg_sigma,
            vmax=hduw.bkg_median + 3 * hduw.bkg_sigma
        )
        self.fig.colorbar(self.im, ax=self.ax)

        self._pan_press = None
        self._selector: Optional[TrailSelector] = None
        self._awaiting_annulus_click = False  # if True, next left-click sets annulus centre (key 'O')
        self._hint = None

        # Events
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press) 

        # ---------- SRCLIST cache (for nearest-neighbour selection) ----------
        self._srclist_path: Optional[Path] = None
        self._srclist_ra: Optional[np.ndarray] = None
        self._srclist_dec: Optional[np.ndarray] = None
        self._srclist_x: Optional[np.ndarray] = None
        self._srclist_y: Optional[np.ndarray] = None

        # ---------- Calibration (apcorr) interaction state ----------
        self._calib_arm = False               # user pressed 'A' and now must press 1..5
        self._calib_active = False            # currently in calibration slot workflow
        self._calib_slot: Optional[int] = None
        self._calib_width: Optional[float] = None   # adjustable length for star box
        self._calib_center: Optional[Tuple[float, float]] = None
        self._calib_srclist_index: Optional[int] = None

        # Store star selections: slot -> dict with star + geometry
        self.calib_star_selections: Dict[int, Dict[str, Any]] = {}


    def add_sources(
        self,
        sources=None,
        pos1=None,
        pos2=None,
        wcs=None,
        *,
        store_key: str = "_marker_artists",
        clear_previous: bool = True,
        **style,
    ):

        """
        Pinta marcadores de fuentes sobre self.ax.
        Devuelve lista de artistas creados.
        """
        ax = getattr(self, "ax", None)
        if ax is None:
            print("[UI] WARN: no hay self.ax; no se pueden dibujar fuentes.")
            return []

        W = wcs or getattr(self, "wcs", None)
        # En WCSAxes, forzar que (x,y) se interprete como pixeles.
        pix_transform = None
        try:
            if hasattr(ax, "get_transform"):
                pix_transform = ax.get_transform("pixel")
        except Exception:
            pix_transform = None


        # Estilo visible por defecto
        base_style = dict(
            marker="o",
            s=30,
            facecolors="none",
            edgecolors="blue",
            linewidths=0.8,
            alpha=0.95,
            zorder=10,
        )

        # Si el usuario pasa 'color', úsalo como edgecolors si no se indicó
        if "color" in style and "edgecolors" not in style and "ec" not in style:
            base_style["edgecolors"] = style.pop("color")
        base_style.update(style or {})

        # Estilos específicos para pos1/pos2
        pos1_style = dict(
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.6,
            alpha=0.98,
            zorder=12,
        )
        pos2_style = dict(
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.6,
            alpha=0.98,
            zorder=12,
        )

        xs, ys = [], []

        def _push_xy(x, y):
            try:
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(float(x))
                    ys.append(float(y))
            except Exception:
                pass

        def _world_to_pixel(ra, dec):
            if W is None:
                return None
            try:
                sc = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
                x, y = W.world_to_pixel(sc)
                x = float(np.atleast_1d(x)[0])
                y = float(np.atleast_1d(y)[0])

                if not (np.isfinite(x) and np.isfinite(y)):
                    print(f"[UI] WARN: RA/Dec→pix devolvió no finito: ra={ra} dec={dec} -> x={x} y={y}")
                    return None

                return x, y
            except Exception as e:
                print(f"[UI] WARN: fallo RA/Dec→pix: {e}")
                return None


        def _pt_to_xy(pt):
            """Devuelve (x,y) en píxel o None."""
            if pt is None or not isinstance(pt, (tuple, list)) or len(pt) != 2:
                return None
            a, b = pt

            # Heurística: si parece RA/Dec y hay WCS, convierto
            if (
                W is not None
                and isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and (-360.0 <= a <= 360.0)
                and (-90.0 <= b <= 90.0)
            ):
                return _world_to_pixel(a, b)

            # Si no, asumo píxel
            try:
                return float(a), float(b)
            except Exception:
                return None

        # ---- 1) sources ----
        try:
            if sources is not None:
                if isinstance(sources, dict):
                    if "xcentroid" in sources and "ycentroid" in sources:
                        xarr = np.atleast_1d(sources["xcentroid"])
                        yarr = np.atleast_1d(sources["ycentroid"])
                        for X, Y in zip(xarr, yarr):
                            _push_xy(X, Y)
                    elif "ra" in sources and "dec" in sources:
                        ra_arr = np.atleast_1d(sources["ra"])
                        dec_arr = np.atleast_1d(sources["dec"])
                        for ra, dec in zip(ra_arr, dec_arr):
                            xy = _pt_to_xy((ra, dec))
                            if xy is not None:
                                _push_xy(*xy)
                    else:
                        # ¿pos1/pos2 dentro del dict?
                        if "pos1" in sources:
                            xy = _pt_to_xy(sources["pos1"])
                            if xy is not None:
                                _push_xy(*xy)
                        if "pos2" in sources:
                            xy = _pt_to_xy(sources["pos2"])
                            if xy is not None:
                                _push_xy(*xy)

                        # O algún iterable de puntos dentro de values()
                        for v in sources.values():
                            try:
                                for pt in v:
                                    xy = _pt_to_xy(pt)
                                    if xy is not None:
                                        _push_xy(*xy)
                            except Exception:
                                continue
                else:
                    # Iterable de puntos
                    for pt in sources:
                        xy = _pt_to_xy(pt)
                        if xy is not None:
                            _push_xy(*xy)
        except Exception as e:
            print(f"[UI] WARN: no pude interpretar 'sources': {e}")

        # ---- 2) dibujado ----
        arts = []

        # # Limpia marcadores previos (evita stacking)
        # if not hasattr(self, "_marker_artists"):
        #     self._marker_artists = []
        # for a in self._marker_artists:
        #     try:
        #         a.remove()
        #     except Exception:
        #         pass
        # self._marker_artists = []

        # Limpia artistas previos SOLO de la "capa" indicada (evita stacking sin borrar otras capas)
        if not hasattr(self, store_key):
            setattr(self, store_key, [])
        if clear_previous:
            for a in getattr(self, store_key, []):
                try:
                    a.remove()
                except Exception:
                    pass
            setattr(self, store_key, [])


        # Dibuja sources (si hay)
        if len(xs) > 0:
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()           
            if pix_transform is not None and "transform" not in base_style:
                base_style = dict(base_style)
                base_style["transform"] = pix_transform
            
            scat = ax.scatter(xs, ys, **base_style)
            arts.append(scat)
            ax.set_xlim(xlim0)
            ax.set_ylim(ylim0)


        # Dibuja pos1/pos2 con estilo propio (si hay)
        # xy1 = _pt_to_xy(pos1)
        # if xy1 is not None:
        #     x1, y1 = xy1
        #     arts.append(ax.scatter([x1], [y1], **pos1_style))

        # xy2 = _pt_to_xy(pos2)
        # if xy2 is not None:
        #     x2, y2 = xy2
        #     arts.append(ax.scatter([x2], [y2], **pos2_style))

                # Dibuja pos1/pos2 con estilo propio (si hay)
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()

        xy1 = _pt_to_xy(pos1)
        if xy1 is not None and np.isfinite(xy1[0]) and np.isfinite(xy1[1]):
            x1, y1 = xy1
            print(f"[UI][DBG] pos1 pix=({x1:.2f},{y1:.2f})")
            print(f"[UI][DBG] xlim={ax.get_xlim()} ylim={ax.get_ylim()}")

            # arts.append(ax.scatter([x1], [y1], **pos1_style))
            s1 = dict(pos1_style)
            if pix_transform is not None and "transform" not in s1:
                s1["transform"] = pix_transform
            arts.append(ax.scatter([x1], [y1], **s1))



            ax.set_xlim(xlim0); ax.set_ylim(ylim0)
        elif pos1 is not None:
            print(f"[UI] WARN: pos1 no válido/convertible: {pos1} (WCS={'OK' if W is not None else 'None'})")

        # xy2 = _pt_to_xy(pos2)
        # if xy2 is not None and np.isfinite(xy2[0]) and np.isfinite(xy2[1]):
        #     x2, y2 = xy2
        #     print(f"[UI][DBG] pos2 pix=({x2:.2f},{y2:.2f})")
        #     arts.append(ax.scatter([x2], [y2], **pos2_style))
        #     ax.set_xlim(xlim0); ax.set_ylim(ylim0)
        xy2 = _pt_to_xy(pos2)
        if xy2 is not None and np.isfinite(xy2[0]) and np.isfinite(xy2[1]):
            x2, y2 = xy2
            print(f"[UI][DBG] pos2 pix=({x2:.2f},{y2:.2f})")

            s2 = dict(pos2_style)
            if pix_transform is not None and "transform" not in s2:
                s2["transform"] = pix_transform
            arts.append(ax.scatter([x2], [y2], **s2))

            ax.set_xlim(xlim0); ax.set_ylim(ylim0)
        elif pos2 is not None:
            print(f"[UI] WARN: pos2 no válido/convertible: {pos2} (WCS={'OK' if W is not None else 'None'})")

        elif pos2 is not None:
            print(f"[UI] WARN: pos2 no válido/convertible: {pos2} (WCS={'OK' if W is not None else 'None'})")

        if len(arts) == 0:
            print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")
            return []

        # self._marker_artists = arts
        setattr(self, store_key, arts)

        try:
            self.update()
        except Exception:
            ax.figure.canvas.draw_idle()

        return arts

    def add_markers(self, ra1, dec1, ra2, dec2, wcs=None, **style):
        try:
            pos1 = (float(ra1), float(dec1))
            pos2 = (float(ra2), float(dec2))
        except Exception:
            print("[UI] WARN: add_markers recibió RA/Dec no numéricos; no pinto marcadores.")
            return

        W = wcs or getattr(self, "wcs", None)
        if W is None:
            # Ojo: si tu helper está dentro de la clase, usa self.extract_wcs_from_hduw(...)
            try:
                W = extract_wcs_from_hduw(self.hduw)
            except Exception:
                W = None
        # En WCSAxes, fuerza que los (x,y) que pasamos a scatter sean píxeles.
        pix_transform = None
        try:
            if hasattr(ax, "get_transform"):
                pix_transform = ax.get_transform("pixel")
        except Exception:
            pix_transform = None
        
        self.add_sources(
            pos1=pos1,
            pos2=pos2,
            wcs=W,
            store_key="_marker_artists",
            clear_previous=True,
            **style,
        )


    def select_trail(self, selector):
        self._selector = selector
        self._draw_hint_text()

        # Run until Enter (done) or Esc (canceled).
        # Your key handler should set selector.finalize() on Enter and selector.canceled=True on Esc.
        import matplotlib.pyplot as plt
        while not (selector.done or selector.canceled):
            plt.pause(0.03)

        if selector.done:
            self._clear_hint_text()
            # Export selection in canonical form and return it
            sel = selector.as_dict()
            self.selection = sel
            self.update()  # one last refresh if you like
            return sel

        # canceled → nothing selected
        self.selection = {"aperture": None, "annulus": None}
        return self.selection

    def update(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # -------------------- helpers --------------------
    def _draw_hint_text(self, text=None):
        # Remove old hint first
        if getattr(self, "_hint", None) is not None:
            try:
                self._hint.remove()
            except Exception:
                pass
            self._hint = None

        default = (
            "Left click: start / set end\n"
            "Right-drag: pan | Wheel: zoom\n"
            "[/]: height  { / }: annulus\n"
            "O: outside background (right-click)\n"
            "I: reset background   R: reset   Enter: accept   Esc: cancel"
        )
        msg = default if text is None else text

        self._hint = self.ax.text(
            0.02, 0.98, msg,
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.3, pad=3),
            zorder=1000,
        )
        self.update()


    def _clear_hint_text(self):
        if getattr(self, "_hint", None) is not None:
            try:
                self._hint.remove()
            except Exception:
                pass
        self._hint = None
        self.update()

    def _hard_reset_selector(self, sel):
        """
        Fully reset selector state so nothing from the previous selection can redraw.
        """
        # Clear selection endpoints and flags
        sel._start = None
        sel._end = None
        sel.done = False

        # Clear derived geometry
        sel.centre = None
        sel.width = None
        sel.theta = None
        sel.rectangular_aperture = None
        sel.rectangular_annulus = None

        # Clear any external background override
        if hasattr(sel, "reset_annulus_centre"):
            try:
                sel.reset_annulus_centre()
            except Exception:
                pass

        # Remove preview patches from axes
        self._remove_preview()

        # Ensure mouse move doesn't keep updating old state
        self._selection_frozen = False


    def _ensure_on_axes(self, event) -> bool:
        return (event.inaxes == self.ax) and (event.xdata is not None) and (event.ydata is not None)

    # -------------------- zoom & pan --------------------
    def _on_scroll(self, event):
        if not self._ensure_on_axes(event): return
        base = 1.2
        scale = (1 / base) if event.button == 'up' else base
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata

        w = (cur_xlim[1] - cur_xlim[0]) * scale
        h = (cur_ylim[1] - cur_ylim[0]) * scale

        rx = (x - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        ry = (y - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([x - rx * w, x + (1 - rx) * w])
        self.ax.set_ylim([y - ry * h, y + (1 - ry) * h])
        self.update()

    def _on_mouse_press(self, event):
        if not self._ensure_on_axes(event):
            return
        if self._selector is None:
            return

        sel = self._selector

        # 0) If calibration slot is active: left-click selects nearest SRCLIST source
        if self._calib_active and event.button == 1 and event.xdata is not None and event.ydata is not None:
            hit = self._nearest_srclist(event.xdata, event.ydata, max_dist_pix=10.0)
            if hit is None:
                print("[UI] WARN: No SRCLIST source near click.")
                return
            j, xs, ys, ra, dec = hit
            self._calib_srclist_index = j
            self._calib_center = (xs, ys)
            print(f"[UI] Selected SRCLIST source: idx={j}  RA={ra:.6f} Dec={dec:.6f}  x={xs:.2f} y={ys:.2f}")
            self._draw_calib_preview(self._selector)
            return

        # 1) OUTSIDE MODE: right-click places background box (consume click; do NOT pan)
        if self._outside_mode and event.button == 3:
            self._outside_mode = False
            self._clear_hint_text()
            try:
                sel.start_background_box(event.xdata, event.ydata)
                self._draw_preview(sel)
            except Exception as e:
                print(f"[UI] WARN: failed to place outside background box: {e}")
            return

        # 2) Normal right button: start pan
        if event.button == 3:
            self._pan_press = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            return


        # 3) Left click: trail selection
        if event.button != 1:
            return

        # If selection is frozen, a left-click starts a new selection (common UX)
        if self._selection_frozen:
            self._selection_frozen = False
            sel._start = None
            sel._end = None

        if sel._start is None:
            # First click: start point
            sel._start = (event.xdata, event.ydata)
            sel._end = (event.xdata, event.ydata)
            sel._compute_from_points(sel._start, sel._end)
            self._draw_preview(sel)
        else:
            # Second click: end point and FREEZE geometry (do not keep following mouse)
            sel._end = (event.xdata, event.ydata)
            sel._compute_from_points(sel._start, sel._end)
            self._draw_preview(sel)
            self._selection_frozen = True

    def _on_mouse_release(self, event):
        if event.button == 3:
            self._pan_press = None

    def _on_mouse_move(self, event):
        # Panning
        if self._pan_press is not None and event.inaxes == self.ax:
            x0, y0, (x1, x2), (y1, y2) = self._pan_press
            inv = self.ax.transData.inverted()
            x0d, y0d = inv.transform((x0, y0))
            x1d, y1d = inv.transform((event.x, event.y))
            ddx, ddy = (x1d - x0d), (y1d - y0d)
            self.ax.set_xlim(x1 - ddx, x2 - ddx)
            self.ax.set_ylim(y1 - ddy, y2 - ddy)
            self.update()
            return

        # Live preview while moving the mouse (if start is set and not frozen)
        if self._selector is None:
            return
        sel = self._selector
        if sel.done:
            return
        if getattr(self, "_selection_frozen", False):
            return
        if sel._start is None or sel._end is None:
             return

    def _on_key_press(self, event):
        if self._selector is None:
            return
        sel = self._selector

        # Section for SRCLIST apcorr selection
            # ---------- Calibration mode arming: press A then 1..5 ----------
        if event.key in ("a", "A"):
            self._calib_arm = True
            self._calib_active = False
            self._calib_slot = None
            self._draw_hint_text("Calibration (apcorr) mode:\n"
                                "Press a digit 1..5 to choose slot (A1..A5), then LEFT-click near a yellow source.\n"
                                "←/→: adjust box length   [ ]: adjust height   { }: adjust annulus\n"
                                "Enter: save slot   R: reset slot")
            return

        if self._calib_arm and event.key in ("1", "2", "3", "4", "5"):
            self._calib_slot = int(event.key)
            self._calib_arm = False
            self._calib_active = True

            # start with trail geometry (same height / annulus / theta / width if available)
            if sel.width is None or not np.isfinite(sel.width):
                self._calib_width = 30.0
            else:
                self._calib_width = float(sel.width)

            self._calib_center = None
            self._calib_srclist_index = None

            self._draw_hint_text(f"Calibration slot A{self._calib_slot} active:\n"
                                "LEFT-click near a yellow source to place the same box/annulus centered on that star.\n"
                                "←/→: adjust box length   [ ]: adjust height   { }: adjust annulus\n"
                                "Enter: save slot   R: reset slot")
            return

        # ---------- While calibration slot is active ----------
        if self._calib_active:
            # Arrow keys adjust LENGTH (width) of the star box
            k = (event.key or "").lower()

            LEFT_KEYS = {"left", "shift+left", "ctrl+left", "alt+left"}
            RIGHT_KEYS = {"right", "shift+right", "ctrl+right", "alt+right"}

            if k in LEFT_KEYS:
                if self._calib_width is not None:
                    self._calib_width = max(5.0, float(self._calib_width) - 1.0)
                    self._safe_redraw_preserve_view(lambda: self._draw_calib_preview(sel))
                return

            if k in RIGHT_KEYS:
                if self._calib_width is not None:
                    self._calib_width = float(self._calib_width) + 1.0
                    self._safe_redraw_preserve_view(lambda: self._draw_calib_preview(sel))
                return

            # Reset only this slot
            if event.key in ("r", "R"):
                if self._calib_slot in self.calib_star_selections:
                    del self.calib_star_selections[self._calib_slot]
                self._calib_center = None
                self._calib_srclist_index = None
                self._remove_preview()  # removes current preview (calib box)
                self._draw_hint_text(f"Calibration slot A{self._calib_slot} reset.\n"
                                    "LEFT-click near a yellow source to select again.")
                return

            # Save slot on Enter (do NOT close figure)
            if event.key == "enter":
                if self._calib_slot is None or self._calib_center is None or self._calib_srclist_index is None:
                    print("[UI] WARN: Calibration slot not ready (no star selected).")
                    return

                cx, cy = self._calib_center
                self.calib_star_selections[self._calib_slot] = dict(
                    slot=self._calib_slot,
                    srclist_index=self._calib_srclist_index,
                    x=cx, y=cy,
                    width=float(self._calib_width),
                    height=float(sel.height),
                    semi_out=float(sel.semi_out),
                    theta=float(sel.theta) if sel.theta is not None else 0.0,
                    ra=float(self._srclist_ra[self._calib_srclist_index]) if self._srclist_ra is not None else np.nan,
                    dec=float(self._srclist_dec[self._calib_srclist_index]) if self._srclist_dec is not None else np.nan,
                    srclist_file=str(self._srclist_path) if self._srclist_path else "",
                )

                print(f"[UI] Calibration saved: A{self._calib_slot}  idx={self._calib_srclist_index}  "
                    f"x={cx:.2f} y={cy:.2f}  width={self._calib_width:.1f} height={sel.height:.1f} semi_out={sel.semi_out:.1f}")

                # Exit calibration slot (user can press A then next digit)
                self._calib_active = False
                self._calib_slot = None
                self._calib_center = None
                self._calib_srclist_index = None
                self._clear_hint_text()
                # Optionally redraw trail preview
                try:
                    self._draw_preview(sel)
                except Exception:
                    pass
                return



        if event.key == 'escape':
            sel.canceled = True
            plt.close(self.fig)
            return

        if event.key == 'enter':
            # Finalize ONLY on Enter
            if sel._start is not None and sel._end is not None:
                sel.done = True
                plt.close(self.fig)
            return

        # Toggle outside-background placement
        if event.key and event.key.lower() == "o":
            # Only meaningful once a trail exists
            if sel._start is not None and sel._end is not None:
                self._outside_mode = True
                self._draw_hint_text("Outside background mode: RIGHT-click to define background box\n\n"
                                    "Then use { } to resize annulus as usual\n"
                                    "I: reset background to follow trail")
            return

        # Reset background to follow trail
        if event.key in ('i', 'I'):
            self._outside_mode = False
            self._clear_hint_text()
            try:
                sel.reset_annulus_centre()
                self._draw_preview(sel)
            except Exception:
                pass
            return

        if event.key in ("r", "R"):
            self._outside_mode = False
            self._selection_frozen = False

            # Clear selector state
            sel._start = None
            sel._end = None
            sel.done = False

            # Clear derived geometry so preview can't redraw
            sel.centre = None
            sel.width = None
            sel.theta = None
            sel.rectangular_aperture = None
            sel.rectangular_annulus = None

            # Reset background placement
            try:
                sel.reset_annulus_centre()
            except Exception:
                pass

            # Remove existing drawings
            self._remove_preview()

            # Re-draw the default hint (single instance)
            self._draw_hint_text()

            self.update()
            return



        # Geometry adjustments
        if event.key == '[':
            sel.adjust_height(-1.0); self._draw_preview(sel)
        elif event.key == ']':
            sel.adjust_height(+1.0); self._draw_preview(sel)
        elif event.key == '{':   # Shift + [
            sel.adjust_annulus(-1.0); self._draw_preview(sel)
        elif event.key == '}':   # Shift + ]
            sel.adjust_annulus(+1.0); self._draw_preview(sel)

    # -------------------- preview drawing --------------------

    def _remove_preview(self):
        # UI-level
        for attr in ("_ap_patch", "_an_patch"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # Selector-level (in case older code stores them there)
        sel = getattr(self, "_selector", None)
        if sel is not None:
            for attr in ("_ap_patch", "_an_patch"):
                artist = getattr(sel, attr, None)
                if artist is not None:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                    setattr(sel, attr, None)

    def _draw_preview(self, sel):
        # Always remove previous preview before drawing a new one
        self._remove_preview()

        # If no geometry, nothing to draw
        if sel is None or sel.rectangular_aperture is None or sel.rectangular_annulus is None:
            self.update()
            return

        # Prefer filled patches via _to_patch (so we control alpha etc.)
        try:
            ap_patch = sel.rectangular_aperture._to_patch(
                fill=True, facecolor='tab:blue', alpha=0.35, edgecolor='k', lw=0.7
            )
            an_patch = sel.rectangular_annulus._to_patch(
                fill=True, facecolor='tab:red', alpha=0.15, edgecolor='none'
            )

            self.ax.add_patch(ap_patch)
            self.ax.add_patch(an_patch)

            # Store references so _remove_preview can remove them
            self._ap_patch = ap_patch
            self._an_patch = an_patch
            sel._ap_patch = ap_patch
            sel._an_patch = an_patch

        except Exception:
            # Fall back to photutils plot() but store references so we can remove later
            ap_artists = sel.rectangular_aperture.plot(ax=self.ax, color='tab:blue', lw=1.0)
            an_artists = sel.rectangular_annulus.plot(ax=self.ax, color='tab:red', lw=0.8)

            # photutils typically returns a list; keep the first artist
            self._ap_patch = ap_artists[0] if isinstance(ap_artists, (list, tuple)) and ap_artists else ap_artists
            self._an_patch = an_artists[0] if isinstance(an_artists, (list, tuple)) and an_artists else an_artists
            sel._ap_patch = self._ap_patch
            sel._an_patch = self._an_patch

        self.update()


    def add_srclist_overlay(
        self,
        srclist_path: Path,
        *,
        wcs=None,
        marker: str = ".",
        s: float = 20,
        alpha: float = 0.8,
        color: str = "yellow",
        zorder: int = 11,
        coi_frac_max: float = 0.02,
        min_points_after_coi: int = 5,
    ) -> List:
        """Overplot SRCLIST sources as yellow points.

        Strategy:
        1) Read SRCLIST table columns (RA_CORR/DEC_CORR preferred, else RA/DEC; also RATE/CORR_RATE).
        2) Apply COI filter where possible: |(CORR_RATE - RATE)/RATE| < coi_frac_max and RATE>0.
            If too few points remain, fall back to plotting without COI filtering.
        3) Convert world -> pixel using the WCS of the currently displayed image (typically FSIMAG).
            This avoids the mismatch that occurs if using XPOS/YPOS measured on a different grid (e.g. FIMAG).
        4) Cache pixel coordinates + SRCLIST rates for nearest-neighbour selection later.
        """
        srclist_path = Path(srclist_path)

        try:
            tab = self._read_srclist_table(srclist_path)

            # Prefer corrected coordinates if present
            ra = tab.get("RA_CORR")
            dec = tab.get("DEC_CORR")
            if ra is None or dec is None:
                ra = tab.get("RA")
                dec = tab.get("DEC")

            if ra is None or dec is None:
                raise KeyError("SRCLIST table missing RA/DEC (or RA_CORR/DEC_CORR) columns")

            ra = np.asarray(ra, dtype=float)
            dec = np.asarray(dec, dtype=float)

            rate = tab.get("RATE")
            corr_rate = tab.get("CORR_RATE")

            # Base mask: finite sky coords
            mask = np.isfinite(ra) & np.isfinite(dec)

            # COI filter (optional + robust fallback)
            if rate is not None and corr_rate is not None:
                rate = np.asarray(rate, dtype=float)
                corr_rate = np.asarray(corr_rate, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = (corr_rate - rate) / rate

                mask_coi = (
                    np.isfinite(frac) &
                    np.isfinite(rate) &
                    (rate > 0) &
                    (np.abs(frac) < coi_frac_max)
                )

                # Apply only if enough points remain
                if np.count_nonzero(mask & mask_coi) >= min_points_after_coi:
                    mask &= mask_coi
                else:
                    print(f"[UI] WARN: COI filter removed almost all sources; overlaying without COI filter for file ({srclist_path})")

            # Need WCS to plot correctly over FSIMAG
            W = wcs or getattr(self, "wcs", None)
            if W is None:
                raise ValueError("No WCS available to project SRCLIST RA/Dec onto image pixels")

            sc = SkyCoord(ra[mask] * u.deg, dec[mask] * u.deg, frame="icrs")
            x, y = W.world_to_pixel(sc)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            # Pixel validity mask (after projection)
            mxy = np.isfinite(x) & np.isfinite(y)
            x = x[mxy]
            y = y[mxy]

            # Cache for nearest-neighbour and later lookup
            # We must apply the same mxy selection to all cached arrays.
            # Build index array of the rows that survived sky-mask, then filter by mxy.
            idx_rows = np.nonzero(mask)[0]
            idx_rows = idx_rows[mxy]

            self._srclist_path = srclist_path
            self._srclist_x = x
            self._srclist_y = y

            rate_all = tab.get("RATE")
            rateerr_all = tab.get("RATE_ERR")
            corrrate_all = tab.get("CORR_RATE")
            srcnum_all = tab.get("SRCNUM")
            qflag_all = tab.get("QFLAG")
            cflag_all = tab.get("CFLAG")
            eflag_all = tab.get("EFLAG")
            sig_all = tab.get("SIGNIFICANCE")

            self._srclist_rate = np.asarray(rate_all, dtype=float)[idx_rows] if rate_all is not None else None
            self._srclist_rate_err = np.asarray(rateerr_all, dtype=float)[idx_rows] if rateerr_all is not None else None
            self._srclist_corr_rate = np.asarray(corrrate_all, dtype=float)[idx_rows] if corrrate_all is not None else None
            self._srclist_srcnum = np.asarray(srcnum_all)[idx_rows] if srcnum_all is not None else None
            self._srclist_qflag = np.asarray(qflag_all)[idx_rows] if qflag_all is not None else None
            self._srclist_cflag = np.asarray(cflag_all)[idx_rows] if cflag_all is not None else None
            self._srclist_eflag = np.asarray(eflag_all)[idx_rows] if eflag_all is not None else None
            self._srclist_significance = np.asarray(sig_all, dtype=float)[idx_rows] if sig_all is not None else None

            # Also keep sky coords for reporting
            self._srclist_ra = ra[idx_rows]
            self._srclist_dec = dec[idx_rows]

        except Exception as e:
            print(f"[UI] WARN: could not load SRCLIST overlay ({srclist_path}): {e}")
            return []

        # Plot points in pixel coords; dict form avoids RA/Dec heuristics in add_sources
        return self.add_sources(
            sources={"xcentroid": self._srclist_x, "ycentroid": self._srclist_y},
            store_key="_srclist_artists",      
            clear_previous=True,               
            marker=marker,
            s=s,
            facecolors=color,
            edgecolors=color,
            linewidths=0,
            alpha=alpha,
            zorder=zorder,
        )



    def _nearest_srclist(self, x: float, y: float, max_dist_pix: float = 8.0):
        """Return (idx, x_s, y_s, ra_s, dec_s) of nearest SRCLIST source to (x,y)."""
        if self._srclist_x is None or self._srclist_y is None:
            return None
        dx = self._srclist_x - float(x)
        dy = self._srclist_y - float(y)
        d2 = dx*dx + dy*dy
        j = int(np.nanargmin(d2))
        if not np.isfinite(d2[j]) or np.sqrt(d2[j]) > max_dist_pix:
            return None
        return (j, float(self._srclist_x[j]), float(self._srclist_y[j]),
                float(self._srclist_ra[j]), float(self._srclist_dec[j]))

    def _draw_calib_preview(self, sel):
        """Draw preview for calibration-star box using current selector geometry + calib center/width."""
        if self._calib_center is None or self._calib_width is None:
            return
        cx, cy = self._calib_center
        theta = float(sel.theta) if sel.theta is not None else 0.0

        # Build a "fake" selector-like object with the required attributes
        class _Tmp:
            pass
        tmp = _Tmp()
        tmp.height = float(sel.height)
        tmp.semi_out = float(sel.semi_out)
        tmp.theta = theta
        tmp.centre = (cx, cy)
        tmp.width = float(self._calib_width)

        # Use TrailSelector's build helpers if you have them; if not, rely on existing preview code:
        # Here we assume your preview code uses tmp.centre/tmp.width/tmp.height/tmp.theta/tmp.semi_out
        # and creates tmp.rectangular_aperture/tmp.rectangular_annulus. If not, tell me and I adapt.

        try:
            # If your TrailSelector already has a method to compute rectangles from centre/width/theta:
            sel._compute_geometry_from_centre_width(tmp.centre, tmp.width, tmp.theta)  # only if exists
        except Exception:
            pass

        # Minimal: set up the photutils apertures directly if your preview expects those
        tmp.rectangular_aperture = RectangularAperture(tmp.centre, w=tmp.width, h=tmp.height, theta=tmp.theta)
        tmp.rectangular_annulus = RectangularAnnulus(tmp.centre, w_in=tmp.width, h_in=tmp.height,
                                                    w_out=tmp.width + 2*tmp.semi_out,
                                                    h_out=tmp.height + 2*tmp.semi_out,
                                                    theta=tmp.theta)

        self._draw_preview(tmp)

    @staticmethod
    def _read_srclist_table(srclist_path: Path) -> dict:
        """Read key columns from an OM SWSRLI source list table."""
        srclist_path = Path(srclist_path)
        if not srclist_path.exists():
            raise FileNotFoundError(f"SRCLIST not found: {srclist_path}")

        with fits.open(str(srclist_path), memmap=False) as hdul:
            tab = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                if getattr(hdu, "columns", None) is not None:
                    tab = hdu.data
                    break
            if tab is None:
                raise ValueError(f"No table HDU found in SRCLIST: {srclist_path.name}")

            names = tab.columns.names
            up = {n.upper(): n for n in names}

            def get(name: str):
                k = up.get(name.upper())
                return np.asarray(tab[k]) if k else None

            return {
                "SRCNUM": get("SRCNUM"),
                "RA_CORR": get("RA_CORR"),
                "DEC_CORR": get("DEC_CORR"),
                "RATE": get("RATE"),
                "RATE_ERR": get("RATE_ERR"),
                "CORR_RATE": get("CORR_RATE"),
                "CORR_RATE_ERR": get("CORR_RATE_ERR"),
                "BACKGROUND_RATE": get("BACKGROUND_RATE"),
                "BKG_RATE_ERR": get("BKG_RATE_ERR"),
                "QFLAG": get("QFLAG"),
                "CFLAG": get("CFLAG"),
                "EFLAG": get("EFLAG"),
            }

    def _safe_redraw_preserve_view(self, redraw_fn):
        """Redraw without changing zoom/pan state or axes limits."""
        ax = self.ax
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()

        # Disable toolbar interactive modes (zoom/pan) to avoid drag_zoom state issues
        tb = getattr(getattr(self.fig, "canvas", None), "toolbar", None)
        if tb is not None:
            mode = getattr(tb, "mode", "")
            if mode:
                try:
                    m = str(mode).lower()
                    if "zoom" in m:
                        tb.zoom()   # toggle off
                    elif "pan" in m:
                        tb.pan()    # toggle off
                    else:
                        tb.mode = ""
                except Exception:
                    pass

        try:
            redraw_fn()
        finally:
            ax.set_xlim(xlim0)
            ax.set_ylim(ylim0)
            ax.figure.canvas.draw_idle()
