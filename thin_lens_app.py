# thin_lens_app.py
# (Moved from 'thin lens android app.py' for a valid Python module name)

# optic_sim_pillow_vectorized_fixed.py
# Kivy + Pillow + NumPy vectorised lens simulator (Real & Virtual images)
# Dependencies: kivy, pillow, numpy

import os
import time
import numpy as np
from PIL import Image, ImageDraw
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from threading import Thread, Lock
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.uix.popup import Popup
from kivy.logger import Logger
from kivy.graphics.texture import Texture
from kivy.utils import platform

# Using optic.kv from the same directory; no in-file KV fallback needed anymore.

class FileChooserPopup(Popup):
    select_cb = None
    def select(self, path, selection):
        if selection:
            filepath = selection[0]
            if self.select_cb:
                self.select_cb(filepath)
            self.dismiss()

class MainUI(BoxLayout):
    current_png = StringProperty('')
    focal = NumericProperty(1.2)
    obj_width = NumericProperty(0.8)
    obj_x = NumericProperty(1.5)  # object distance (do) measured from lens at x=0 toward +x
    show_grid = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_array = None
        self.img_pil = None
        self.img_fname = None
        # obj_x is a NumericProperty; keep only y as plain attribute
        self.obj_y = 0.0
        self._touching = False
        self.render_pending = False
        self.debug = False
        self._render_thread = None
        self._render_lock = Lock()
        self._cancel_render = False
        try:
            app_dir = App.get_running_app().user_data_dir  # type: ignore
        except Exception:
            app_dir = os.getcwd()
        os.makedirs(app_dir, exist_ok=True)
        self._storage_dir = app_dir
        self._tmp_png = os.path.join(self._storage_dir, 'tmp_optic_render.png')
        # initial blank
        blank = Image.new('RGBA', (800, 480), (245, 245, 250, 255))
        blank.save(self._tmp_png)
        self.current_png = self._tmp_png
        self._last_render = None
        self._init_sample_image()
        Clock.schedule_once(self._bind_size_events, 0)
        # Paint a tiny sanity texture immediately (helps diagnose blank-canvas issues on Android)
        Clock.schedule_once(lambda dt: self._paint_sanity(), 0)
        Clock.schedule_once(self._first_render, 0.12)

    def _bind_size_events(self, *_):
        try:
            self.ids.png_view.bind(
                size=lambda *a: self.render_figure_debounced(),
                pos=lambda *a: self.render_figure_debounced()
            )
        except Exception:
            pass

    def _first_render(self, dt):
        self.render_figure()

    def _paint_sanity(self):
        try:
            # 8x8 red/green checker to prove the widget shows textures before first render
            arr = np.zeros((8, 8, 4), dtype=np.uint8)
            arr[:, :, 0] = 220
            arr[::2, ::2, 1] = 160
            arr[1::2, 1::2, 1] = 80
            arr[:, :, 3] = 255
            sanity = Image.fromarray(arr, 'RGBA')
            self._update_texture_from_pil(sanity)
        except Exception as e:
            Logger.debug(f"OpticSim: sanity texture failed: {e}")

    def open_filechooser(self):
        popup = FileChooserPopup()
        popup.select_cb = self.load_image
        popup.open()

    def _init_sample_image(self):
        if self.img_pil is not None:
            return
        w, h = 240, 240
        sample = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(sample)
        for r in range(min(w, h)//2, 0, -4):
            c = int(255 * r / (min(w, h)//2))
            d.ellipse((w/2 - r, h/2 - r, w/2 + r, h/2 + r), fill=(30, 30, 120, int(160 * (r/(min(w,h)//2))) ))
        arrow_color = (255, 180, 40, 255)
        d.polygon([
            (w*0.15, h*0.85), (w*0.15, h*0.15), (w*0.45, h*0.15),
            (w*0.45, h*0.05), (w*0.70, h*0.20), (w*0.45, h*0.35), (w*0.45, h*0.25), (w*0.25, h*0.25), (w*0.25, h*0.85)
        ], fill=arrow_color)
        self.img_pil = sample
        self.img_array = np.array(sample)
        self.img_fname = 'built_in_sample'

    def load_image(self, filepath):
        # On Android, FileChooser returns real file paths; but if we ever receive a content:// URI, resolve it.
        if isinstance(filepath, str) and filepath.startswith('content://'):
            resolved = self._resolve_android_uri(filepath)
            if resolved:
                filepath = resolved
            else:
                Logger.warning(f"OpticSim: Unable to resolve Android URI: {filepath}")
                return
        try:
            im = Image.open(filepath).convert('RGBA')
        except Exception as e:
            print("Error loading:", e)
            return
        maxdim = 480
        try:
            resamp = Image.Resampling.LANCZOS  # Pillow >= 9.1
        except Exception:
            resamp = getattr(Image, 'LANCZOS', Image.BICUBIC)
        try:
            im.thumbnail((maxdim, maxdim), resamp)
        except Exception:
            # final fallback
            im.thumbnail((maxdim, maxdim))
        self.img_pil = im
        self.img_array = np.array(im)
        self.img_fname = filepath
        self.obj_x = 1.5
        self.obj_y = 0.0
        self.render_figure()

    def _resolve_android_uri(self, uri):
        """Best-effort copy of a content:// URI to a local temp file; return file path or None."""
        try:
            if platform != 'android':
                return None
            if not (isinstance(uri, str) and uri.startswith('content://')):
                return None
            import importlib
            jnius = importlib.import_module('jnius')  # type: ignore
            autoclass = jnius.autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            content_resolver = activity.getContentResolver()
            Uri = autoclass('android.net.Uri')
            input_stream = content_resolver.openInputStream(Uri.parse(uri))
            if input_stream is None:
                return None
            # Choose a destination path
            dest = os.path.join(self._storage_dir, 'picked_image')
            # Try to detect extension from mime
            MimeTypeMap = autoclass('android.webkit.MimeTypeMap')
            mime = content_resolver.getType(Uri.parse(uri))
            ext = None
            if mime:
                ext = '.' + str(MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) or '').strip('.')
            if not ext or len(ext) > 5:
                ext = '.png'
            dest = dest + ext
            # Copy stream
            with open(dest, 'wb') as fh:
                buf = bytearray(64 * 1024)
                while True:
                    # read returns Java int; -1 for EOF
                    n = input_stream.read(buf)
                    if n == -1:
                        break
                    if n > 0:
                        fh.write(memoryview(buf)[:n])
            input_stream.close()
            Logger.info(f"OpticSim: Copied content URI to {dest}")
            return dest
        except Exception as e:
            Logger.warning(f"OpticSim: URI resolve failed: {e}")
            return None

    def toggle_grid(self, show):
        self.show_grid = show
        self.render_figure()

    def on_focal(self, instance, value):
        try:
            fv = float(value)
        except Exception:
            return
        if hasattr(self, 'ids') and 'focal_label' in self.ids:
            self.ids.focal_label.text = f"f = {fv:.2f}"
        self.render_figure_debounced()

    def on_obj_width(self, instance, value):
        try:
            wv = float(value)
        except Exception:
            return
        if hasattr(self, 'ids') and 'obj_label' in self.ids:
            self.ids.obj_label.text = f"width = {wv:.2f}"
        self.render_figure_debounced()

    def on_obj_x(self, instance, value):
        """Slider updates for object distance (do)."""
        try:
            vx = float(value)
        except Exception:
            return
        if hasattr(self, 'ids') and 'objx_label' in self.ids:
            self.ids.objx_label.text = f"do = {vx:.2f}"
        self.render_figure_debounced()

    def reset_view(self):
        self.obj_x = 1.5
        self.obj_y = 0.0
        self.focal = 1.2
        try:
            self.ids.focal_slider.value = self.focal
            self.ids.obj_slider.value = self.obj_width
            if 'objx_slider' in self.ids:
                self.ids.objx_slider.value = self.obj_x
        except Exception:
            pass
        self.render_figure()

    def export_params(self):
        out = {
            'mode': 'auto',
            'focal': float(self.focal),
            'obj_width': float(self.obj_width),
            'obj_x': float(self.obj_x),
            'obj_y': float(self.obj_y),
            'source_image': self.img_fname or '',
        }
        fn = os.path.join(os.getcwd(), 'optic_params.txt')
        with open(fn, 'w') as f:
            for k,v in out.items():
                f.write(f"{k}: {v}\n")
        from kivy.uix.label import Label
        popup = Popup(title='Exported', content=Label(text=f"Saved:\n{fn}"), size_hint=(0.7, 0.3))
        popup.open()

    def save_current_png(self):
        fn = os.path.join(self._storage_dir, f"optic_snapshot_{int(time.time()*1000)}.png")
        try:
            if self._last_render is not None:
                self._last_render.save(fn)
            elif os.path.exists(self._tmp_png):
                Image.open(self._tmp_png).save(fn)
            else:
                Logger.warning("OpticSim: No render available to save.")
            Logger.info(f"OpticSim: Saved snapshot -> {fn}")
        except Exception as e:
            Logger.warning(f"OpticSim: Save failed: {e}")

    def show_help(self):
        help_txt = ("Controls:\n"
                    "- Open image: load PNG/JPG/TIFF\n"
                    "- Drag on the image to move the object in physical space\n"
                    "- Use sliders to change focal length and object width\n"
                    "- Save PNG to export current view")
        popup = Popup(title='Help', size_hint=(0.9, 0.6))
        from kivy.uix.label import Label
        popup.content = Label(text=help_txt)
        popup.open()

    def render_figure_debounced(self, dt=0.06):
        if self.render_pending:
            return
        self.render_pending = True
        Clock.schedule_once(self._do_render, dt)

    def _do_render(self, dt):
        self.render_pending = False
        self.render_figure()

    def render_figure(self):
        try:
            canvas_w, canvas_h = map(int, self.ids.png_view.size)
        except Exception:
            canvas_w, canvas_h = 1200, 700
        # Clamp extremely large sizes (prevents GPU texture allocation failure -> white)
        max_dim = 2048  # conservative for many mid devices
        if canvas_w > max_dim or canvas_h > max_dim:
            scale = min(max_dim / canvas_w, max_dim / canvas_h)
            canvas_w = int(canvas_w * scale)
            canvas_h = int(canvas_h * scale)
        canvas_w = max(300, canvas_w)
        canvas_h = max(200, canvas_h)
        with self._render_lock:
            self._cancel_render = True
        def _background():
            with self._render_lock:
                self._cancel_render = False
            pil_img = self._do_compose_image(canvas_w, canvas_h)
            with self._render_lock:
                if self._cancel_render:
                    return
            Clock.schedule_once(lambda dt: self._finalize_render(pil_img), 0)
        self._render_thread = Thread(target=_background, daemon=True)
        self._render_thread.start()

    def _do_compose_image(self, canvas_w, canvas_h):
        bg = (248, 249, 255, 255)
        img = Image.new('RGBA', (canvas_w, canvas_h), bg)
        draw = ImageDraw.Draw(img, 'RGBA')
        PLOT_XLIM = (-8.0, 12.0)
        PLOT_YLIM = (-5.0, 5.0)
        def phys_to_px(x, y):
            x0, x1 = PLOT_XLIM
            y0, y1 = PLOT_YLIM
            px = int((x - x0) / (x1 - x0) * canvas_w)
            py = int((1.0 - (y - y0) / (y1 - y0)) * canvas_h)
            return px, py
        if self.show_grid:
            for xi in range(int(PLOT_XLIM[0]), int(PLOT_XLIM[1]) + 1):
                xpx, _ = phys_to_px(xi, PLOT_YLIM[0])
                draw.line([(xpx,0),(xpx,canvas_h)], fill=(225,225,235,255), width=1)
            for yi in range(int(PLOT_YLIM[0]), int(PLOT_YLIM[1]) + 1):
                _, ypx = phys_to_px(PLOT_XLIM[0], yi)
                draw.line([(0,ypx),(canvas_w,ypx)], fill=(225,225,235,255), width=1)
        f = float(self.focal)
        lens_w_phys = 0.2
        lens_h_phys = 3.5
        left_top = phys_to_px(-lens_w_phys/2, lens_h_phys/2)
        right_bot = phys_to_px(lens_w_phys/2, -lens_h_phys/2)
        draw.ellipse([left_top, right_bot], outline=(36,90,255,255), width=3)
        lx1 = phys_to_px(0, -1.75); lx2 = phys_to_px(0, 1.75)
        draw.line([lx1, lx2], fill=(36,90,255,255), width=3)
        for fx in (f, -f):
            px, py = phys_to_px(fx, 0)
            r = 7
            draw.ellipse([(px-r,py-r),(px+r,py+r)], fill=(36,90,255,255))
            draw.text((px-6, py+10), "F", fill=(36,90,255,255))
        if self.img_array is None:
            draw.text((canvas_w//2 - 140, canvas_h//2 - 10), "Open an image to start", fill=(120,120,120,255))
            img.save(self._tmp_png)
            self.current_png = self._tmp_png
            return img
        img_arr = self.img_array
        h_in, w_in = img_arr.shape[:2]
        img_w_phys = float(self.obj_width)
        img_h_phys = img_w_phys * (h_in / w_in)
        x0 = self.obj_x - img_w_phys/2
        x1 = self.obj_x + img_w_phys/2
        y0 = self.obj_y - img_h_phys/2
        y1 = self.obj_y + img_h_phys/2
        p0 = phys_to_px(x0, y1)
        p1 = phys_to_px(x1, y0)
        tw = abs(p1[0] - p0[0]); th = abs(p1[1] - p0[1])
        if tw < 2: tw = 2
        if th < 2: th = 2
        try:
            resamp = Image.Resampling.LANCZOS
        except Exception:
            resamp = getattr(Image, 'LANCZOS', Image.BICUBIC)
        try:
            thumb = self.img_pil.resize((tw, th), resamp)
        except Exception:
            thumb = self.img_pil.resize((tw, th))
        img.paste(thumb, p0, mask=thumb)
        label_px = phys_to_px(self.obj_x, self.obj_y + img_h_phys/2 + 0.15)
        draw.text(label_px, "Object", fill=(200,40,40,255))
        obj_top_y = self.obj_y + img_h_phys/2
        # Thin lens imaging geometry (sign convention: object at +x, real image appears at -x when do>f)
        do = max(1e-6, self.obj_x)
        if abs(do - f) < 1e-6:
            di = None  # image at infinity
            m = None
            image_top_pt = None
        else:
            di = (f * do) / (do - f)
            m = -di / do  # magnification (negative => real inverted)
            # Image center y is magnified version of object center relative to optical axis (y=0)
            img_center_x = -di
            img_center_y = m * self.obj_y
            half_obj_h = img_h_phys / 2.0
            image_half_h = abs(m) * half_obj_h
            img_top_y = img_center_y + image_half_h
            image_top_pt = (img_center_x, img_top_y)
        # Correct Liang-Barsky clipping (fix previous sign mistakes)
        def _clip_line(x0, y0, x1, y1):
            xmin, xmax = PLOT_XLIM; ymin, ymax = PLOT_YLIM
            dx = x1 - x0; dy = y1 - y0
            p = [-dx, dx, -dy, dy]
            q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
            u1, u2 = 0.0, 1.0
            for pi, qi in zip(p, q):
                if abs(pi) < 1e-12:
                    if qi > 0:  # outside and parallel
                        return None
                    continue
                r = qi / pi
                if pi < 0:  # entering
                    if r > u2: return None
                    if r > u1: u1 = r
                else:       # leaving
                    if r < u1: return None
                    if r < u2: u2 = r
            nx0 = x0 + u1 * dx; ny0 = y0 + u1 * dy
            nx1 = x0 + u2 * dx; ny1 = y0 + u2 * dy
            return nx0, ny0, nx1, ny1

        def draw_line_phys(x1, y1, x2, y2, color, width=2):
            clipped = _clip_line(x1, y1, x2, y2)
            if clipped is None:
                return
            ax0, ay0, ax1, ay1 = clipped
            p1 = phys_to_px(ax0, ay0); p2 = phys_to_px(ax1, ay1)
            draw.line([p1, p2], fill=color, width=width)

        def draw_dashed_phys(x1, y1, x2, y2, color, width=2, dlen=0.4):
            clipped = _clip_line(x1, y1, x2, y2)
            if clipped is None:
                return
            ax0, ay0, ax1, ay1 = clipped
            dist = ((ax1 - ax0)**2 + (ay1 - ay0)**2) ** 0.5
            if dist < 1e-9:
                return
            n = max(1, int(dist / dlen))
            for i in range(0, n, 2):
                t0 = i / n; t1 = min((i + 1) / n, 1.0)
                xa = ax0 + (ax1 - ax0) * t0; ya = ay0 + (ay1 - ay0) * t0
                xb = ax0 + (ax1 - ax0) * t1; yb = ay0 + (ay1 - ay0) * t1
                p1 = phys_to_px(xa, ya); p2 = phys_to_px(xb, yb)
                draw.line([p1, p2], fill=color, width=width)
        # Ray 1: object top to lens (parallel ray before lens)
        draw_line_phys(do, obj_top_y, 0, obj_top_y, (220, 40, 40, 255))
        # Ensure horizontal incident segment always visible (fallback direct draw bypassing advanced clip)
        if PLOT_YLIM[0] <= obj_top_y <= PLOT_YLIM[1]:
            p_obj = phys_to_px(do, obj_top_y)
            p_lens = phys_to_px(0, obj_top_y)
            draw.line([p_obj, p_lens], fill=(220, 40, 40, 255), width=2)
        # After lens: through far focal point (-f,0) for real ray (regardless of real/virtual)
        # Extend far beyond plot so clipping finds correct exit boundary.
        ray1_dir_x = -f - 0
        ray1_dir_y = 0 - obj_top_y
        k_far = 20  # large multiplier
        far_x = 0 + k_far * ray1_dir_x
        far_y = obj_top_y + k_far * ray1_dir_y
        draw_line_phys(0, obj_top_y, far_x, far_y, (220, 40, 40, 255))
        if di is not None and di < 0 and image_top_pt is not None:
            # back-trace dashed for virtual image to apparent image point
            draw_dashed_phys(0, obj_top_y, image_top_pt[0], image_top_pt[1], (220, 40, 40, 160))

        # Ray 2: through lens center (0,0) — straight line unaffected
        center_dir_x = 0 - do
        center_dir_y = 0 - obj_top_y
        far2_x = do + 20 * center_dir_x
        far2_y = obj_top_y + 20 * center_dir_y
        draw_line_phys(do, obj_top_y, far2_x, far2_y, (40, 140, 240, 255))
        if di is not None and di < 0 and image_top_pt is not None:
            draw_dashed_phys(0, 0, image_top_pt[0], image_top_pt[1], (40, 140, 240, 160))
        # Physically accurate full-plane inverse mapping transformation
        if di is None:
            # Image at infinity indication
            px_inf1 = phys_to_px(-f*3, self.obj_y)
            px_inf2 = phys_to_px(-f*5, self.obj_y)
            draw.line([px_inf1, px_inf2], fill=(20,120,20,180), width=2)
            draw.text((px_inf1[0], px_inf1[1]-20), 'Image at infinity', fill=(10,120,10,255))
        else:
            # Decide which side to render (real image left, virtual right)
            if di > 0:
                X_min, X_max = PLOT_XLIM[0], 0.0
            else:
                X_min, X_max = 0.0, PLOT_XLIM[1]
            # Pixel region bounds in canvas for that physical span
            px_left, py_top = phys_to_px(X_min, PLOT_YLIM[1])
            px_right, py_bottom = phys_to_px(X_max, PLOT_YLIM[0])
            region_w = max(2, abs(px_right - px_left))
            region_h = max(2, abs(py_bottom - py_top))
            # Build physical coordinate grid covering that side
            X_phys = np.linspace(X_min, X_max, region_w)
            # Top (row 0) corresponds to Y = PLOT_YLIM[1]
            Y_phys = np.linspace(PLOT_YLIM[1], PLOT_YLIM[0], region_h)
            X_grid, Y_grid = np.meshgrid(X_phys, Y_phys)

            # Inverse mapping to object plane (thin lens)
            with np.errstate(divide='ignore', invalid='ignore'):
                x_src = np.where(np.abs(X_grid + f) < 1e-9, np.inf, (X_grid * f) / (X_grid + f))
                y_src = np.where(np.abs(X_grid) < 1e-9, np.inf, (Y_grid * x_src) / X_grid)
            # Object physical extents
            obj_w = img_w_phys
            obj_h = img_h_phys
            # Map to source image pixel coordinates
            w_in, h_in = self.img_pil.size
            # Convert physical to pixel indices (centered at object center)
            x_pix = ((x_src - self.obj_x) / obj_w + 0.5) * w_in
            y_pix = ((self.obj_y - y_src) / obj_h + 0.5) * h_in
            # Vectorized bilinear sampling
            out_rgba = np.zeros((region_h, region_w, 4), dtype=np.uint8)
            valid = (np.isfinite(x_pix) & np.isfinite(y_pix) &
                     (x_pix >= 0) & (x_pix < w_in - 1) &
                     (y_pix >= 0) & (y_pix < h_in - 1))
            if np.any(valid):
                x0 = np.floor(x_pix[valid]).astype(int)
                y0 = np.floor(y_pix[valid]).astype(int)
                dx = x_pix[valid] - x0
                dy = y_pix[valid] - y0
                x1 = x0 + 1
                y1 = y0 + 1
                # Clamp
                x1 = np.clip(x1, 0, w_in - 1)
                y1 = np.clip(y1, 0, h_in - 1)
                src = self.img_array  # (h_in, w_in, 4), already RGBA
                c00 = src[y0, x0].astype(np.float32)
                c10 = src[y0, x1].astype(np.float32)
                c01 = src[y1, x0].astype(np.float32)
                c11 = src[y1, x1].astype(np.float32)
                c0 = c00 * (1 - dx)[:, None] + c10 * dx[:, None]
                c1 = c01 * (1 - dx)[:, None] + c11 * dx[:, None]
                c = c0 * (1 - dy)[:, None] + c1 * dy[:, None]
                out_flat = out_rgba.reshape(-1, 4)
                out_flat[np.flatnonzero(valid), :] = np.clip(c, 0, 255).astype(np.uint8)
            # Create PIL region and paste
            region_img = Image.fromarray(out_rgba, 'RGBA')
            # For real images (m<0) the inversion is already encoded in mapping; for virtual keep as is
            img.paste(region_img, (min(px_left, px_right), 0), mask=region_img)
            # Annotation
            try:
                di_txt = f"di={di:.2f}"
                label = 'Real' if di > 0 else 'Virtual'
                draw.text(phys_to_px(-f*2.2, PLOT_YLIM[0]+0.5), f"do={do:.2f}  {di_txt}  m={m:.2f} ({label})", fill=(50,50,50,255))
            except Exception:
                pass
        return img

    def _finalize_render(self, pil_img):
        self._last_render = pil_img
        self._update_texture_from_pil(pil_img)

    def _update_texture_from_pil(self, pil_img):
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        w, h = pil_img.size
        try:
            tex = Texture.create(size=(w, h), colorfmt='rgba')

            # Feed PIL's top-to-bottom bytes and flip once via Kivy
            buf = pil_img.tobytes()
            tex.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
            tex.flip_vertical()  # keep only this flip

            img_widget = self.ids.get('png_view')
            if img_widget:
                img_widget.texture = tex
                img_widget.canvas.ask_update()
                img_widget.source = ''
        except Exception as e:
            Logger.error(f"OpticSim: Texture update failed: {e}")
            try:
                # Fallback to scaled half-size texture if original fails
                if w > 64 and h > 64:
                    fallback = pil_img.resize((w//2, h//2), Image.Resampling.BILINEAR)
                    self._update_texture_from_pil(fallback)
            except Exception:
                pass

    def on_touch_down(self, touch):
        # Ignore touches on the right-side control panel if present
        ctrl_panel = self.ids.get('control_panel')
        if ctrl_panel and ctrl_panel.collide_point(*touch.pos):
             return super().on_touch_down(touch)
        img = self.ids.get('png_view', None)
        if img and img.collide_point(*touch.pos):
            self._touching = True
            self._update_obj_from_touch(touch)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._touching:
            self._update_obj_from_touch(touch)
            return True
        img = self.ids.get('png_view', None)
        if img and img.collide_point(*touch.pos):
            self._touching = True
            self._update_obj_from_touch(touch)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._touching:
            self._touching = False
            self.render_figure_debounced(0.02)
            return True
        return super().on_touch_up(touch)

    def _update_obj_from_touch(self, touch):
        img_widget = self.ids.png_view
        px, py = img_widget.pos
        pw, ph = img_widget.size
        if self.debug:
            try:
                Logger.debug(f"OpticSim: touch.pos={touch.pos} img.pos={px,py} img.size={pw,ph}")
            except Exception:
                pass
        if not (px <= touch.x <= px + pw and py <= touch.y <= py + ph):
            if self.debug:
                Logger.debug("OpticSim: touch outside image bounds")
            return
        nx = (touch.x - px) / pw
        ny = (touch.y - py) / ph
        PLOT_XLIM = (-8.0, 12.0)
        PLOT_YLIM = (-5.0, 5.0)
        phys_x = PLOT_XLIM[0] + nx * (PLOT_XLIM[1] - PLOT_XLIM[0])
        # Kivy touch Y grows upward, same as our physical +Y; no flip here.
        phys_y = PLOT_YLIM[0] + ny * (PLOT_YLIM[1] - PLOT_YLIM[0])
        min_x = self.obj_width/2
        max_x = PLOT_XLIM[1] - self.obj_width/2
        new_x = float(np.clip(phys_x, min_x, max_x))
        new_y = float(np.clip(phys_y, PLOT_YLIM[0] + 0.05, PLOT_YLIM[1] - 0.05))
        if self.debug:
            Logger.debug(f"OpticSim: phys click -> ({phys_x:.3f}, {phys_y:.3f}) clamped -> ({new_x:.3f}, {new_y:.3f})")
        self.obj_x = new_x
        self.obj_y = new_y
        self.render_figure_debounced(0.02)

class OpticSimApp(App):
    def build(self):
        kv_path = os.path.join(os.path.dirname(__file__), 'optic.kv')
        if os.path.exists(kv_path):
            Builder.load_file(kv_path)
        else:
            Logger.error("OpticSim: 'optic.kv' not found. Please add the file to run the app.")
        Logger.info("OpticSim: Starting build v0.1.1")
        return MainUI()

    def on_stop(self):
        # Gracefully stop any background render
        root = getattr(self, 'root', None)
        try:
            if root and getattr(root, '_render_thread', None) and root._render_thread.is_alive():
                with root._render_lock:
                    root._cancel_render = True
                root._render_thread.join(timeout=0.5)
        except Exception:
            pass

if __name__ == '__main__':
    OpticSimApp().run()
