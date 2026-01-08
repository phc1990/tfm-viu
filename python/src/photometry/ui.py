"""
Improved User Interface (UI) for trail selection with zoom/pan and live preview.
This version DOES NOT finalize on left click; only Enter finalizes.
"""
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
    
from typing import Optional
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

    # def add_sources(
    #     self,
    #     sources=None,
    #     pos1=None,
    #     pos2=None,
    #     wcs=None,
    #     **style,
    # ):
    #     """
    #     Pinta marcadores de fuentes sobre self.ax.

    #     Acepta:
    #     - sources:
    #         * dict con {'xcentroid','ycentroid'}  (píxeles)
    #         * dict con {'ra','dec'}               (grados; requiere WCS)
    #         * iterable de (x,y) en píxeles
    #         * iterable de (ra,dec) en grados (requiere WCS)
    #         * dict con {'pos1':(ra,dec)| (x,y), 'pos2':(...)}
    #     - pos1, pos2: tuplas (ra,dec) en grados o (x,y) en píxeles.
    #     - wcs: astropy.wcs.WCS para RA/Dec→píxeles; si None usa self.wcs.
    #     - style: kwargs para scatter (color, s, etc.)

    #     Devuelve lista de artistas creados.
    #     """
    #     ax = getattr(self, "ax", None)
    #     if ax is None:
    #         print("[UI] WARN: no hay self.ax; no se pueden dibujar fuentes.")
    #         return []

    #     W = wcs or getattr(self, "wcs", None)

    #     # Estilo visible por defecto
    #     base_style = dict(marker='o', s=30, facecolor='none', edgecolor='blue',
    #                     linewidths=0.8, alpha=0.95, zorder=10)
    #     # Si el usuario pasa 'color', úsalo como edgecolor si no se indicó
    #     if "color" in style and "edgecolor" not in style and "ec" not in style:
    #         base_style["edgecolor"] = style.pop("color")
    #     base_style.update(style or {})

    #     # Estilos específicos para pos1/pos2
    #     pos1_style = dict(
    #         marker="o",
    #         s=90,                 
    #         facecolors="none",
    #         edgecolors="tab:green",
    #         linewidths=1.6,       
    #         alpha=0.98,
    #         zorder=12,
    #     )
    #     pos2_style = dict(
    #         marker="o",
    #         s=90,
    #         facecolors="none",
    #         edgecolors="tab:red",
    #         linewidths=1.6,
    #         alpha=0.98,
    #         zorder=12,
    #     )

    #     xs, ys = [], []

    #     def _push_xy(x, y):
    #         try:
    #             if np.isfinite(x) and np.isfinite(y):
    #                 xs.append(float(x)); ys.append(float(y))
    #         except Exception:
    #             pass

    #     def _world_to_pixel_list(ra_arr, dec_arr):
    #         if W is None:
    #             print("[UI] WARN: RA/Dec sin WCS; omito esas fuentes.")
    #             return
    #         try:
    #             sc = SkyCoord(np.atleast_1d(ra_arr) * u.deg, np.atleast_1d(dec_arr) * u.deg, frame="icrs")
    #             X, Y = W.world_to_pixel(sc)
    #             for Xk, Yk in zip(np.atleast_1d(X), np.atleast_1d(Y)):
    #                 _push_xy(Xk, Yk)
    #         except Exception as e:
    #             print(f"[UI] WARN: fallo RA/Dec→pix: {e}")

    #     def _add_point(pt):
    #         if pt is None or not isinstance(pt, (tuple, list)) or len(pt) != 2:
    #             return
    #         a, b = pt
    #         # Heurística: si parece RA/Dec y hay WCS, convierto
    #         if W is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)) \
    #         and (-360.0 <= a <= 360.0) and (-90.0 <= b <= 90.0):
    #             _world_to_pixel_list([a], [b])
    #         else:
    #             _push_xy(a, b)

    #     # ---- 1) sources ----
    #     try:
    #         if sources is not None:
    #             if isinstance(sources, dict):
    #                 # Posibles alias
    #                 if "xcentroid" in sources and "ycentroid" in sources:
    #                     xarr = np.atleast_1d(sources["xcentroid"])
    #                     yarr = np.atleast_1d(sources["ycentroid"])
    #                     for X, Y in zip(xarr, yarr):
    #                         _push_xy(X, Y)
    #                 elif "ra" in sources and "dec" in sources:
    #                     _world_to_pixel_list(sources["ra"], sources["dec"])
    #                 else:
    #                     # ¿pos1/pos2 dentro del dict?
    #                     if "pos1" in sources: _add_point(sources["pos1"])
    #                     if "pos2" in sources: _add_point(sources["pos2"])
    #                     # O algún iterable de puntos en el dict
    #                     for v in sources.values():
    #                         try:
    #                             for pt in v:
    #                                 _add_point(pt)
    #                         except Exception:
    #                             continue
    #             else:
    #                 # Iterable de puntos (x,y) o (ra,dec)
    #                 for pt in sources:
    #                     _add_point(pt)
    #     except Exception as e:
    #         print(f"[UI] WARN: no pude interpretar 'sources': {e}")
        
    #     # ---- 2) pos1 / pos2 ----
    #         _add_point(pos1)
    #         _add_point(pos2)

    #         arts = []

    #         # Convertir puntos por separado para poder aplicar estilos distintos
    #         p1 = None
    #         p2 = None

    #     if pos1 is not None:
    #         p1 = _add_point(pos1)
    #         x1, y1 = pos1
    #         scat1 = ax.scatter([x1], [y1], **pos1_style)
    #         arts.append(scat1)
    #     if pos2 is not None:
    #         p2 = _add_point(pos2)
    #         x2, y2 = p2
    #         scat2 = ax.scatter([x2], [y2], **pos2_style)
    #         arts.append(scat2)

    #     # Si no hay puntos válidos, informar
    #     if len(arts) == 0:
    #         print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")
    #         return

    #     # Guardar artistas para poder eliminarlos luego (evita stacking)
    #     if not hasattr(self, "_marker_artists"):
    #         self._marker_artists = []

    #     # Eliminar marcadores previos
    #     for a in self._marker_artists:
    #         try:
    #             a.remove()
    #         except Exception:
    #             pass
    #     self._marker_artists = arts

    #     # Redibujar
    #     try:
    #         self.update()
    #     except Exception:
    #         ax.figure.canvas.draw_idle()

        


    #     # arts = []
    #     # if len(xs) > 0:
    #     #     scat = ax.scatter(xs, ys, **base_style)
    #     #     arts.append(scat)
    #     #     try:
    #     #         # Si hay método update, úsalo; si no, dibuja
    #     #         self.update()
    #     #     except Exception:
    #     #         ax.figure.canvas.draw_idle()
    #     # else:
    #     #     print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")

    #     # # Guarda artistas (opcional) para poder limpiarlos luego si quieres
    #     # if not hasattr(self, "_source_artists"):
    #     #     self._source_artists = []
    #     # self._source_artists.extend(arts)

    #     # return arts

    def add_sources(
        self,
        sources=None,
        pos1=None,
        pos2=None,
        wcs=None,
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
                return float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0])
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

        # Limpia marcadores previos (evita stacking)
        if not hasattr(self, "_marker_artists"):
            self._marker_artists = []
        for a in self._marker_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._marker_artists = []

        # Dibuja sources (si hay)
        if len(xs) > 0:
            scat = ax.scatter(xs, ys, **base_style)
            arts.append(scat)

        # Dibuja pos1/pos2 con estilo propio (si hay)
        xy1 = _pt_to_xy(pos1)
        if xy1 is not None:
            x1, y1 = xy1
            arts.append(ax.scatter([x1], [y1], **pos1_style))

        xy2 = _pt_to_xy(pos2)
        if xy2 is not None:
            x2, y2 = xy2
            arts.append(ax.scatter([x2], [y2], **pos2_style))

        if len(arts) == 0:
            print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")
            return []

        self._marker_artists = arts

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

        self.add_sources(pos1=pos1, pos2=pos2, wcs=W, **style)

    # def add_markers(self, *args, **kwargs):
    #     """Backward-compatible alias for add_sources (used by other pipeline modules)."""
    #     return self.add_sources(*args, **kwargs)

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
    # def _draw_hint_text(self, text=None):
    #     # Remove old hint if present
    #     if self._hint is not None:
    #         try:
    #             self._hint.remove()
    #         except Exception:
    #             pass
    #         self._hint = None

    #     default = (
    #         "Left click: start / set end (no finalize)\n"
    #         "Wheel: zoom | Right-drag: pan\n"
    #         "[/]: height  { / }: annulus  R: reset\n"
    #         "Enter: accept & close   Esc: cancel"
    #     )

    #     msg = default if text is None else text

    #     self._hint = self.ax.text(
    #         0.02, 0.98, msg,
    #         transform=self.ax.transAxes, va='top', ha='left',
    #         color='cyan', fontsize=8,
    #         bbox=dict(facecolor='black', alpha=0.3, pad=3),
    #         zorder=50
    #     )
    #     self.update()

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


    # def _clear_hint_text(self):
    #     if self._hint is not None:
    #         try:
    #             self._hint.remove()
    #         except Exception:
    #             pass
    #     self._hint = None
    #     self.update()

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

        # # 3) Left click: trail selection
        # if event.button != 1:
        #     return

        # if sel._start is None:
        #     sel._start = (event.xdata, event.ydata)
        #     sel._end = (event.xdata, event.ydata)
        #     sel._compute_from_points(sel._start, sel._end)
        #     self._draw_preview(sel)
        # else:
        #     sel._end = (event.xdata, event.ydata)
        #     sel._compute_from_points(sel._start, sel._end)
        #     self._draw_preview(sel)
        #     if sel.finalize_on_click:
        #         sel.done = True


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

        # if self._selector is None or self._selector.done:
        #     return
        # sel = self._selector
        # if sel._start is not None and self._ensure_on_axes(event):
        #     sel._end = (event.xdata, event.ydata)
        #     sel._compute_from_points(sel._start, sel._end)
        #     self._draw_preview(sel)

    def _on_key_press(self, event):
        if self._selector is None:
            return
        sel = self._selector

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

        # if event.key in ('r', 'R'):
        #     # reset selection + any outside background override
        #     self._outside_mode = False
        #     self._clear_hint_text()
        #     try:
        #         sel.reset_annulus_centre()
        #     except Exception:
        #         pass

        #     sel._start, sel._end, sel.done = None, None, False
        #     self._remove_preview()
        #     self.update()
        #     return

        # # Outside background mode toggle
        # if event.key and event.key.lower() == "o":
        #     # Require trail to be defined before allowing outside background placement
        #     if sel._start is not None and sel._end is not None:
        #         self._outside_mode = True
        #         self._draw_hint_text("Outside background mode: RIGHT-click to define background box")
        #     return

        # # Reset background to follow trail
        # if event.key in ('i', 'I'):
        #     self._outside_mode = False
        #     self._clear_hint_text()
        #     try:
        #         sel.reset_annulus_centre()
        #         self._draw_preview(sel)
        #     except Exception:
        #         pass
        #     return
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
