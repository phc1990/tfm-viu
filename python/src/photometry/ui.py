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
    def __init__(self, height: float = 5.0, semi_out: float = 5.0, finalize_on_click: bool = False) -> None:
        # User-adjustable geometry
        self.height = float(height)
        self.semi_out = float(semi_out)

        # Behavior flag (default False -> Enter is required to finalize)
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
        self.rectangular_aperture: Optional[RectangularAperture] = None
        self.rectangular_annulus: Optional[RectangularAnnulus] = None

        # Preview patches
        self._ap_patch = None
        self._an_patch = None

    def _compute_from_points(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = (x2 - x1), (y2 - y1)
        width = float(np.hypot(dx, dy))
        if width == 0:
            return

        centre = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
        theta = np.arctan2(dy, dx)

        self.centre = centre
        self.width = width
        self.theta = theta

        self.rectangular_aperture = RectangularAperture(
            positions=self.centre, w=self.width, h=self.height, theta=self.theta
        )
        self.rectangular_annulus = RectangularAnnulus(
            positions=self.centre,
            w_in=self.width, w_out=self.width + 2 * self.semi_out,
            h_in=self.height, h_out=self.height + 2 * self.semi_out,
            theta=self.theta
        )

    def is_done(self) -> bool:
        return bool(self.done)

    def adjust_height(self, delta: float):
        self.height = max(1.0, self.height + float(delta))
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    def adjust_annulus(self, delta: float):
        self.semi_out = max(0.5, self.semi_out + float(delta))
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)

    # --- inside TrailSelector class ---

    @property
    def aperture(self):
        """Compatibility alias for photometry.py: returns rectangular_aperture."""
        return self.rectangular_aperture

    @property
    def annulus(self):
        """Compatibility alias for photometry.py: returns rectangular_annulus."""
        return self.rectangular_annulus

    def as_dict(self):
        """Return selection with canonical keys used by the pipeline."""
        return {
            "aperture": self.rectangular_aperture,
            "annulus": self.rectangular_annulus,
        }
    
    def finalize(self):
        # compute geometry if both points exist and not yet computed
        if self._start is not None and self._end is not None:
            self._compute_from_points(self._start, self._end)
        self.done = True


    def set_wcs(self, wcs):
        self.wcs = wcs


class UI:
    def __init__(self, hduw: HDUW) -> None:
        plt.ion()
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
    #     _add_point(pos1)
    #     _add_point(pos2)

    #     arts = []
    #     if len(xs) > 0:
    #         scat = ax.scatter(xs, ys, **base_style)
    #         arts.append(scat)
    #         try:
    #             # Si hay método update, úsalo; si no, dibuja
    #             self.update()
    #         except Exception:
    #             ax.figure.canvas.draw_idle()
    #     else:
    #         print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")

    #     # Guarda artistas (opcional) para poder limpiarlos luego si quieres
    #     if not hasattr(self, "_source_artists"):
    #         self._source_artists = []
    #     self._source_artists.extend(arts)

    #     return arts
    

    def add_sources(
        self,
        sources=None,
        pos1=None,
        pos2=None,
        wcs=None,
        **style,
    ):
        ax = getattr(self, "ax", None)
        if ax is None:
            print("[UI] WARN: no hay self.ax; no se pueden dibujar fuentes.")
            return []

        W = wcs or getattr(self, "wcs", None)

        # Estilo por defecto (sources)
        src_style = dict(
            marker="o",
            s=30,                 # tamaño (área) del círculo
            facecolors="none",
            edgecolors="blue",
            linewidths=0.8,
            alpha=0.95,
            zorder=10,
        )
        # Permite que el usuario ajuste el estilo de sources con kwargs
        if "color" in style and "edgecolor" not in style and "ec" not in style:
            src_style["edgecolors"] = style.pop("color")
        src_style.update(style or {})

        # Estilos específicos para pos1/pos2
        pos1_style = dict(
            marker="o",
            s=90,                 # más grande
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.6,       # línea más gruesa
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

        def _convert_point(pt):
            """Convierte (ra,dec) o (x,y) a píxeles. Devuelve (x,y) o None."""
            if pt is None or not isinstance(pt, (tuple, list)) or len(pt) != 2:
                return None
            a, b = pt

            # Heurística: si parece RA/Dec y hay WCS, convierto
            try:
                if (
                    W is not None
                    and isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and (-360.0 <= a <= 360.0) and (-90.0 <= b <= 90.0)
                ):
                    sc = SkyCoord(a * u.deg, b * u.deg, frame="icrs")
                    x, y = W.world_to_pixel(sc)
                else:
                    x, y = a, b

                if np.isfinite(x) and np.isfinite(y):
                    return float(x), float(y)
            except Exception as e:
                print(f"[UI] WARN: fallo al convertir punto {pt}: {e}")
            return None

        src_pts = []   # lista de (x,y) para sources
        pos1_pt = None
        pos2_pt = None

        # ---- 1) sources ----
        try:
            if sources is not None:
                if isinstance(sources, dict):
                    if "xcentroid" in sources and "ycentroid" in sources:
                        xarr = np.atleast_1d(sources["xcentroid"])
                        yarr = np.atleast_1d(sources["ycentroid"])
                        for X, Y in zip(xarr, yarr):
                            if np.isfinite(X) and np.isfinite(Y):
                                src_pts.append((float(X), float(Y)))
                    elif "ra" in sources and "dec" in sources:
                        if W is None:
                            print("[UI] WARN: RA/Dec sin WCS; omito esas fuentes.")
                        else:
                            sc = SkyCoord(np.atleast_1d(sources["ra"]) * u.deg,
                                        np.atleast_1d(sources["dec"]) * u.deg, frame="icrs")
                            X, Y = W.world_to_pixel(sc)
                            for Xk, Yk in zip(np.atleast_1d(X), np.atleast_1d(Y)):
                                if np.isfinite(Xk) and np.isfinite(Yk):
                                    src_pts.append((float(Xk), float(Yk)))
                    else:
                        # Si dentro del dict vienen pos1/pos2, respétalos como pos1/pos2
                        if "pos1" in sources:
                            pos1_pt = _convert_point(sources["pos1"])
                        if "pos2" in sources:
                            pos2_pt = _convert_point(sources["pos2"])

                        # Otros valores: intentar interpretarlos como lista de puntos
                        for v in sources.values():
                            try:
                                for pt in v:
                                    xy = _convert_point(pt)
                                    if xy is not None:
                                        src_pts.append(xy)
                            except Exception:
                                continue
                else:
                    for pt in sources:
                        xy = _convert_point(pt)
                        if xy is not None:
                            src_pts.append(xy)
        except Exception as e:
            print(f"[UI] WARN: no pude interpretar 'sources': {e}")

        # ---- 2) pos1 / pos2 explícitos ----
        # (si ya vinieron en sources como dict, estos pueden sobreescribir si se pasan explícitos)
        if pos1 is not None:
            pos1_pt = _convert_point(pos1)
        if pos2 is not None:
            pos2_pt = _convert_point(pos2)

        arts = []

        # Pinta sources (azul)
        if len(src_pts) > 0:
            xs, ys = zip(*src_pts)
            arts.append(ax.scatter(xs, ys, **src_style))

        # Pinta pos1 (verde) y pos2 (rojo)
        if pos1_pt is not None:
            x1, y1 = pos1_pt
            arts.append(ax.scatter([x1], [y1], **pos1_style))
        if pos2_pt is not None:
            x2, y2 = pos2_pt
            arts.append(ax.scatter([x2], [y2], **pos2_style))

        if len(arts) > 0:
            try:
                self.update()
            except Exception:
                ax.figure.canvas.draw_idle()
        else:
            print("[UI] NOTE: add_sources() no pintó nada (sin puntos válidos).")

        if not hasattr(self, "_source_artists"):
            self._source_artists = []
        self._source_artists.extend(arts)

        return arts


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
    def _draw_hint_text(self):
        if self._hint is not None:
            try: self._hint.remove()
            except Exception: pass
        self._hint = self.ax.text(
            0.02, 0.98,
            "Left click: start / set end (no finalize)\n"
            "Wheel: zoom | Right-drag: pan\n"
            "[/]: height  { / }: annulus  R: reset\n"
            "Enter: accept & close   Esc: cancel",
            transform=self.ax.transAxes, va='top', ha='left',
            color='cyan', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.3, pad=3)
        )

    def _clear_hint_text(self):
        try:
            if self._hint is not None:
                self._hint.remove()
        except Exception:
            pass
        self._hint = None

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
        # Right button: start pan
        if event.button == 3 and self._ensure_on_axes(event):
            self._pan_press = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            return

        # Left click: selection (does NOT finalize)
        if self._selector is None or event.button != 1 or not self._ensure_on_axes(event):
            return

        sel = self._selector
        if sel._start is None:
            sel._start = (event.xdata, event.ydata)
            sel._end = (event.xdata, event.ydata)
            sel._compute_from_points(sel._start, sel._end)
            self._draw_preview(sel)
        else:
            # Update end point; keep preview; do NOT set sel.done here
            sel._end = (event.xdata, event.ydata)
            sel._compute_from_points(sel._start, sel._end)
            self._draw_preview(sel)
            if sel.finalize_on_click:
                sel.done = True  # only if user explicitly enabled legacy behavior

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

        # Live preview while moving the mouse (if start is set and not finalized)
        if self._selector is None or self._selector.done:
            return
        sel = self._selector
        if sel._start is not None and self._ensure_on_axes(event):
            sel._end = (event.xdata, event.ydata)
            sel._compute_from_points(sel._start, sel._end)
            self._draw_preview(sel)

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
        if event.key in ('r', 'R'):
            sel._start, sel._end, sel.done = None, None, False
            self._remove_preview()
            self.update()
            return
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
        for artist in (getattr(self._selector, "_ap_patch", None),
                       getattr(self._selector, "_an_patch", None)):
            try:
                if artist is not None:
                    artist.remove()
            except Exception:
                pass
        self._selector._ap_patch = None
        self._selector._an_patch = None

    def _draw_preview(self, sel: TrailSelector):
        self._remove_preview()
        if sel.rectangular_aperture is None or sel.rectangular_annulus is None:
            self.update(); return

        # Try private filled patch; fall back to outline if needed
        try:
            ap_patch = sel.rectangular_aperture._to_patch(fill=True, color='tab:blue', alpha=0.35, ec='k', lw=0.5)
            an_patch = sel.rectangular_annulus._to_patch(fill=True, color='tab:red',  alpha=0.15, ec='none')
            self.ax.add_patch(ap_patch)
            self.ax.add_patch(an_patch)
            sel._ap_patch = ap_patch
            sel._an_patch = an_patch
        except Exception:
            sel.rectangular_aperture.plot(ax=self.ax, color='tab:blue', lw=1.0)
            sel.rectangular_annulus.plot(ax=self.ax, color='tab:red',  lw=0.8)
        self.update()



    def extract_wcs_from_hduw(hduw):
        """
        Try several ways to get a usable celestial WCS from an HDU wrapper.
        Returns a WCS or None.
        """
        # 1) If the wrapper already has a WCS, use it
        w = getattr(hduw, "wcs", None)
        if w is not None:
            try:
                # ensure it's a real celestial WCS
                if hasattr(w, "has_celestial"):
                    return w if w.has_celestial else None
                return w
            except Exception:
                pass

        # 2) Try the HDU object inside the wrapper
        h = getattr(hduw, "hdu", None)
        if h is not None and hasattr(h, "header"):
            try:
                w = WCS(h.header)
                if not hasattr(w, "has_celestial") or w.has_celestial:
                    return w
            except Exception:
                pass

        # 3) Open the file and search for the first image HDU with a valid WCS
        fpath = str(getattr(hduw, "file", "") or "")
        if fpath:
            try:
                with fits.open(fpath, memmap=False) as hdul:
                    # prefer the same index as hduw.hdu if it’s part of this file
                    if h is not None and hasattr(h, "name"):
                        try:
                            for hh in hdul:
                                if getattr(hh, "name", None) == h.name and hasattr(hh, "header"):
                                    ww = WCS(hh.header)
                                    if not hasattr(ww, "has_celestial") or ww.has_celestial:
                                        return ww
                        except Exception:
                            pass

                    # otherwise, scan all image extensions
                    for hh in hdul:
                        if not hasattr(hh, "data") or hh.data is None:
                            continue
                        try:
                            ww = WCS(hh.header)
                            if not hasattr(ww, "has_celestial") or ww.has_celestial:
                                return ww
                        except Exception:
                            continue

                    # last fallback: primary header
                    try:
                        ww = WCS(hdul[0].header)
                        if not hasattr(ww, "has_celestial") or ww.has_celestial:
                            return ww
                    except Exception:
                        pass
            except Exception:
                pass

        # 4) As a final fallback, try a header attribute on the wrapper
        hdr = getattr(hduw, "header", None)
        if hdr is not None:
            try:
                w = WCS(hdr)
                if not hasattr(w, "has_celestial") or w.has_celestial:
                    return w
            except Exception:
                pass

        return None



    def add_markers(
        self,
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float,   
        wcs: None, 
        **style
    ):
        """
        Compatibility helper for expo_photometry.py.

        Lee pos1/pos2 de los argumentos de línea de comandos:
        - args.pos1_ra, args.pos1_dec
        - args.pos2_ra, args.pos2_dec

        y los pasa a add_sources(), usando WCS si está disponible.
        """
        try:
            pos1 = (ra1, dec1)
            pos2 = (ra2, dec2)
        except AttributeError:
            # Si no tiene esos atributos, no hacemos nada (no rompemos el flujo)
            print("[UI] WARN: args does not have pos1_ra/pos1_dec/pos2_ra/pos2_dec")
            return

        # Usa WCS explícito si se pasa, si no el propio self.wcs
        W = wcs or getattr(self, "wcs", None)

        if W is None:
            W = extract_wcs_from_hduw(self.hduw)
        # add_sources ya sabe manejar (ra,dec) vs (x,y)
        self.add_sources(pos1=pos1, pos2=pos2, wcs=W)
