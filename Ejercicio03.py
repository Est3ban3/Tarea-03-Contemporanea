# animaciones_fisica_contemporanea.py
# ------------------------------------------------------------
# Un solo script con las TRES animaciones:
#   1) Efecto fotoeléctrico (fotón absorbido, emisión si hν>φ)
#   2) Dispersión Compton (elástica-inelástica, p⃗ conservado; γ↓derecha, e↑derecha)
#   3) Dispersión Thomson (elástica, E' = E; electrón ~ en reposo, azul)
#
# Uso típico (Windows PowerShell/CMD):
#   python animaciones_fisica_contemporanea.py --mode foto --save --out efecto_fotoelectrico.mp4
#   python animaciones_fisica_contemporanea.py --mode compton --save --out compton.mp4
#   python animaciones_fisica_contemporanea.py --mode thomson --save --out thomson.mp4
#   python animaciones_fisica_contemporanea.py --mode all --save
#
# Notas de robustez para guardar MP4:
#  - Se usa FIG_SIZE y FIG_DPI para obtener resolución par (640x420) y evitar
#    errores de yuv420p. Filtro ffmpeg scale=trunc(iw/2)*2:trunc(ih/2)*2.
#  - Si --save y no --show: se fuerza backend "Agg" (sin Tkinter).
#  - No se cambia el tamaño de la figura después de creada.
# ------------------------------------------------------------

import argparse
import numpy as np
import matplotlib as mpl

# ------------ Parseo de argumentos ------------
ap = argparse.ArgumentParser(description="Animaciones: Fotoeléctrico, Compton, Thomson")
ap.add_argument("--mode", choices=["foto", "compton", "thomson", "all"], required=True,
                help="Qué animación ejecutar/guardar.")
ap.add_argument("--save", action="store_true", help="Guardar a MP4 (ffmpeg requerido).")
ap.add_argument("--show", action="store_true", help="Mostrar en pantalla.")
ap.add_argument("--out", default=None, help="Archivo de salida MP4 (si --save).")
ap.add_argument("--fps", type=int, default=30, help="FPS de salida (default 30).")
args = ap.parse_args()

# Si vamos a GUARDAR sin mostrar, usar backend sin GUI para evitar errores Tk/TkAgg
if args.save and not args.show:
    mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation

# ------------ Parámetros comunes (pares en píxeles) ------------
FIG_DPI  = 100
FIG_SIZE = (6.4, 4.2)           # 640x420 px (pares)
XMIN, XMAX = -2.4, 4.0
YMIN, YMAX = -2.1, 2.3

# ------------ Constantes físicas útiles ------------
h   = 6.62607015e-34
c   = 299_792_458.0
eC  = 1.602176634e-19
m_e = 9.10938356e-31
mc2 = 511e3 * eC
lambda_C = h/(m_e*c)

# ============================================================
#                    FENÓMENO 1: Fotoeléctrico
# ============================================================
def anim_fotoelectric(fps=30, save=False, show=False, out_path="efecto_fotoelectrico.mp4"):
    # Mini IA (logística) para decidir prob. de emisión en base a (hν, φ)
    def sigmoid(z): return 1.0/(1.0+np.exp(-z))
    def train_logreg(X, y, lr=0.1, epochs=500):
        N, d = X.shape
        W = np.zeros(d); b = 0.0
        for _ in range(epochs):
            p = sigmoid(X@W + b)
            W -= lr*(X.T@(p-y))/N
            b -= lr*np.mean(p-y)
        return W, b
    def dataset(n=2000, seed=7):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(1.0, 6.0, size=n)
        hv  = rng.uniform(0.5, 8.0, size=n)
        y_true = (hv > phi).astype(float)
        flip = (rng.random(n) < 0.05)
        y = np.where(flip, 1.0-y_true, y_true)
        X = np.column_stack([hv, phi, hv-phi])
        return X, y

    # Parámetros físicos editables
    phi_eV = 2.30
    nu = 1.00e15
    phi_J = phi_eV*eC
    E_photon = h*nu
    K = max(E_photon - phi_J, 0.0)
    v_e = np.sqrt(2*K/m_e) if K>0 else 0.0

    # Entrenar IA y decidir emisión
    Xs, ys = dataset()
    W_ai, b_ai = train_logreg(Xs, ys, lr=0.2, epochs=800)
    hv_eV = E_photon/eC
    p_emit = float(sigmoid(np.array([hv_eV, phi_eV, hv_eV-phi_eV])@W_ai + b_ai))
    emit_decision = (p_emit > 0.5) and (K > 0)

    # Escenas/tiempos
    T_HIT = 1.25
    T_TOTAL = 7.0
    FRAMES = int(T_TOTAL * fps)

    v_ref = 1.5e6
    v_anim = (v_e / v_ref)*2.0

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax.set_xlim(-1.6, 5.1); ax.set_ylim(-1.1, 2.6); ax.set_aspect('equal', adjustable='box')
    ax.set_title("Efecto fotoeléctrico — Fotón absorbido y emisión del fotoelectrón")
    surface = plt.Rectangle((0.0, -1.1), 5.2, 3.9, alpha=0.15, ec='k'); ax.add_patch(surface)
    ax.text(0.06, 2.35, "Metal (superficie)", fontsize=9)

    atom_center = np.array([0.60, 0.55])
    ax.plot(atom_center[0], atom_center[1], marker='o', markersize=6)
    orbit_r = 0.26
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(atom_center[0]+orbit_r*np.cos(th), atom_center[1]+orbit_r*np.sin(th), lw=0.6)

    bound_e, = ax.plot([], [], marker='o', markersize=4)
    photon,  = ax.plot([], [], marker='o', markersize=5)
    wave,    = ax.plot([], [], lw=1)
    em_e,    = ax.plot([], [], marker='o', markersize=5)
    txt = ax.text(-1.55, 2.45, "", fontsize=8, va='top', ha='left')

    def set_point(artist, x, y): artist.set_data([x], [y])
    def clear_point(artist): artist.set_data([], [])
    def wave_trail(pos, t, kfreq=18, amp=0.08, L=1.0):
        xs = np.linspace(pos[0]-L, pos[0], 120)
        ys = pos[1] + amp*np.sin(kfreq*xs + 10*t)
        return xs, ys

    X0, Y0 = -1.25, 0.55
    u_in = np.array([1.0, 0.0])

    def init():
        clear_point(photon); wave.set_data([], [])
        clear_point(em_e); clear_point(bound_e); txt.set_text("")
        return photon, wave, em_e, bound_e, txt

    def animate(k):
        t = k / fps
        # Fotón
        if t <= T_HIT:
            pos = np.array([X0, Y0]) + u_in * ((atom_center[0]-X0)/T_HIT)*t
            set_point(photon, pos[0], pos[1])
            xs, ys = wave_trail(pos, t); wave.set_data(xs, ys)
        else:
            clear_point(photon); wave.set_data([], [])

        # Electrón ligado
        if t < T_HIT:
            ang = 5*np.pi*t
            x = atom_center[0] + orbit_r*np.cos(ang)
            y = atom_center[1] + orbit_r*np.sin(ang)
            set_point(bound_e, x, y)
        else:
            clear_point(bound_e)

        # Fotoelectrón emitido
        if emit_decision and t > T_HIT:
            dt = t - T_HIT
            ang = np.deg2rad(20.0)
            v = v_anim
            pos_e = atom_center + np.array([np.cos(ang), np.sin(ang)])*(0.06 + v*dt)
            set_point(em_e, pos_e[0], pos_e[1])
        else:
            clear_point(em_e)

        lines = [
            f"φ = {phi_eV:.2f} eV",
            f"hν = {hv_eV:.2f} eV",
            f"K = {max(hv_eV-phi_eV,0):.2f} eV  ⇒ {'emite e⁻' if emit_decision else 'no hay emisión'}",
        ]
        if K>0 and v_e>0: lines.append(f"v_e ≈ {v_e/1000:.0f} km/s")
        txt.set_text("\n".join(lines))
        return photon, wave, em_e, bound_e, txt

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=FRAMES, interval=1000/fps, blit=True)

    if save:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg no disponible.")
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2000,
            extra_args=["-pix_fmt","yuv420p","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-movflags","+faststart"]
        )
        out = out_path or "efecto_fotoelectrico.mp4"
        anim.save(out, writer=writer, dpi=FIG_DPI)
        print(f"Guardado: {out}")
    if show:
        plt.show()
    plt.close(fig)

# ============================================================
#                    FENÓMENO 2: Compton
# ============================================================
def anim_compton(fps=30, save=False, show=False, out_path="compton.mp4"):
    # Parámetros
    E_gamma_keV = 200.0     # incidente
    theta_deg   = 40.0      # magnitud; fotón disperso se forzará hacia ABAJO
    T_HIT = 1.25
    T_TOTAL = 7.0
    FRAMES = int(T_TOTAL*fps)

    # Cinemática
    E0 = E_gamma_keV*1e3*eC
    p0 = E0/c
    theta = -np.deg2rad(abs(theta_deg))  # hacia abajo
    cosT, sinT = np.cos(theta), np.sin(theta)
    E1 = E0 / (1.0 + (E0/mc2)*(1.0 - cosT))
    p1 = E1 / c

    u_in  = np.array([1.0, 0.0])
    u_out = np.array([np.cos(theta), np.sin(theta)])
    p0v = p0 * u_in
    p1v = p1 * u_out
    pe_vec = p0v - p1v
    # Fuerza electrón al primer cuadrante
    pe_vec = np.array([abs(pe_vec[0]), abs(pe_vec[1])]) + 1e-15
    u_e = pe_vec / np.linalg.norm(pe_vec)

    # Datos para texto
    pe_mag = np.linalg.norm(p0v - p1v)
    E_e_tot = np.sqrt((pe_mag*c)**2 + mc2**2)
    K_e = E_e_tot - mc2
    lam0 = (h*c)/E0
    lam1 = (h*c)/E1
    dlam = lam1 - lam0

    # Velocidades visuales
    X0 = XMIN + 0.2
    v_vis_ph_in = (0.0 - X0)/T_HIT
    v_vis_ph_out = 2.0
    v_vis_e      = 2.0
    ELECTRON_START_OFFSET = 0.15

    # Escena
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect('equal', adjustable='box')
    ax.set_title("Dispersión Compton — γ (↓ derecha) y e⁻ (↑ derecha)")
    ax.axvline(0, lw=0.6); ax.axhline(0, lw=0.6)
    ax.plot(0,0, marker='o', markersize=5, color='0.6', alpha=0.9)
    ax.text(XMIN+0.1, YMAX-0.2, "e⁻ inicial (reposo)", fontsize=8)

    ph_in,    = ax.plot([], [], marker='o', markersize=5)
    trail_in, = ax.plot([], [], lw=1)
    ph_out,   = ax.plot([], [], marker='o', markersize=5)
    trail_out,= ax.plot([], [], lw=1)
    e_out,    = ax.plot([], [], marker='o', markersize=6)
    e_disp_line, = ax.plot([], [], lw=1.2, linestyle='--')

    info = ax.text(XMIN+0.1, YMAX-0.55, "", fontsize=8, va='top', ha='left')

    def set_point(a,x,y): a.set_data([x],[y])
    def clear_point(a): a.set_data([],[])
    def wave_trail(pos, t, kfreq=16, amp=0.09, L=1.2, along_dir=None):
        if along_dir is None: along_dir = np.array([1.0,0.0])
        along_dir = along_dir/np.linalg.norm(along_dir)
        perp = np.array([-along_dir[1], along_dir[0]])
        s = np.linspace(-L, 0.0, 150)
        base = pos + np.outer(s, along_dir)
        wig  = amp*np.sin(kfreq*s + 10*t)
        pts  = base + np.outer(wig, perp)
        return pts[:,0], pts[:,1]

    pos_in_start = np.array([X0, 0.0])

    def init():
        clear_point(ph_in);  trail_in.set_data([],[])
        clear_point(ph_out); trail_out.set_data([],[])
        clear_point(e_out);  e_disp_line.set_data([],[])
        info.set_text("")
        return ph_in, trail_in, ph_out, trail_out, e_out, e_disp_line, info

    def animate(k):
        t = k/fps
        if t <= T_HIT:
            pos = pos_in_start + u_in * v_vis_ph_in * t
            set_point(ph_in, pos[0], pos[1])
            xs, ys = wave_trail(pos, t, along_dir=u_in)
            trail_in.set_data(xs, ys)
            clear_point(ph_out); trail_out.set_data([],[])
            clear_point(e_out);  e_disp_line.set_data([],[])
        else:
            clear_point(ph_in); trail_in.set_data([],[])
            dt = t - T_HIT
            pos_ph = u_out * (0.06 + v_vis_ph_out*dt)
            set_point(ph_out, pos_ph[0], pos_ph[1])
            xs2, ys2 = wave_trail(pos_ph, t, along_dir=u_out)
            trail_out.set_data(xs2, ys2)
            pos_e = u_e * (ELECTRON_START_OFFSET + v_vis_e*dt)
            set_point(e_out, pos_e[0], pos_e[1])
            e_disp_line.set_data([0.0, pos_e[0]], [0.0, pos_e[1]])

        lines = [
            f"Eγ₀ = {E0/eC/1e3:.1f} keV,  Eγ' = {E1/eC/1e3:.1f} keV",
            f"θ_fotón = {np.rad2deg(theta):.1f}° (derecha/abajo)",
            f"u_e = ({u_e[0]:.2f}, {u_e[1]:.2f}) (derecha/arriba)",
            f"Δλ = {dlam*1e12:.3f} pm  (λ_C≈{lambda_C*1e12:.3f} pm)",
            f"K_e ≈ {(K_e/eC)/1e3:.2f} keV",
            f"Offsets/vel: e⁻={ELECTRON_START_OFFSET:.2f}, vγ_out={v_vis_ph_out:.2f}, ve={v_vis_e:.2f}"
        ]
        info.set_text("\n".join(lines))
        return ph_in, trail_in, ph_out, trail_out, e_out, e_disp_line, info

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=FRAMES, interval=1000/fps, blit=True)

    if save:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg no disponible.")
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2000,
            extra_args=["-pix_fmt","yuv420p","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-movflags","+faststart"]
        )
        out = out_path or "compton.mp4"
        anim.save(out, writer=writer, dpi=FIG_DPI)
        print(f"Guardado: {out}")
    if show:
        plt.show()
    plt.close(fig)

# ============================================================
#                    FENÓMENO 3: Thomson
# ============================================================
def anim_thomson(fps=30, save=False, show=False, out_path="thomson.mp4"):
    # Mini IA para muestrear θ ~ sin^2(θ)
    def sigmoid(z): return 1/(1+np.exp(-z))
    def build_dataset(n=4000, seed=7):
        rng = np.random.default_rng(seed)
        theta_deg = rng.uniform(5, 175, size=n)
        theta = np.deg2rad(theta_deg)
        y = np.sin(theta)**2
        y_lbl = (rng.random(n) < y).astype(float)
        X = np.column_stack([np.cos(theta), np.sin(theta), theta/np.pi])
        return X, y_lbl
    def train_logreg(X,y,lr=0.2,epochs=700):
        N,d = X.shape
        W = np.zeros(d); b=0.0
        for _ in range(epochs):
            p = sigmoid(X@W + b)
            W -= lr*(X.T@(p-y))/N
            b -= lr*np.mean(p-y)
        return W,b

    Xtr,ytr = build_dataset()
    W_ai,b_ai = train_logreg(Xtr,ytr)

    rng = np.random.default_rng(10)
    cand_deg = rng.uniform(10, 170, size=200)
    cand = np.deg2rad(cand_deg)
    Xcand = np.column_stack([np.cos(cand), np.sin(cand), cand/np.pi])
    p_sel = sigmoid(Xcand@W_ai + b_ai); p_sel /= (p_sel.sum()+1e-12)
    theta = float(rng.choice(cand, p=p_sel))

    # Parámetros físicos (no cambia energía)
    E_gamma_keV = 25.0
    E0 = E_gamma_keV*1e3*eC
    lam0 = (h*c)/E0

    # Escena/tiempos
    T_HIT = 1.25
    T_TOTAL = 7.0
    FRAMES = int(TOTAL := T_TOTAL*fps)
    u_in = np.array([1.0, 0.0])
    u_out = np.array([np.cos(theta), np.sin(theta)])

    # Velocidades visuales
    X0 = XMIN + 0.2
    v_vis_ph_in  = (0.0 - X0)/T_HIT
    v_vis_ph_out = 2.0

    # Vibración (pequeñísima) del electrón en el origen
    ELECTRON_WIGGLE = 0.08
    DAMPING = 2.5
    WIGGLE_FREQ = 10.0

    # Escena
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect('equal', adjustable='box')
    ax.set_title("Dispersión Thomson — elástica (E' = E), e⁻ ~ en reposo")
    ax.axvline(0, lw=0.6); ax.axhline(0, lw=0.6)
    ax.plot(0,0, marker='o', markersize=6, color='0.6', alpha=0.9)
    ax.text(XMIN+0.1, YMAX-0.2, "e⁻ ~ en reposo", fontsize=8)

    ph_in,     = ax.plot([], [], marker='o', markersize=5)
    trail_in,  = ax.plot([], [], lw=1)
    ph_out,    = ax.plot([], [], marker='o', markersize=5)
    trail_out, = ax.plot([], [], lw=1)
    # Electrón azul (pediste color azul)
    e_vib,     = ax.plot([], [], marker='o', markersize=4, color='blue', alpha=0.9)

    info = ax.text(XMIN+0.1, YMAX-0.55, "", fontsize=8, va='top', ha='left')

    def set_point(a,x,y): a.set_data([x],[y])
    def clear_point(a): a.set_data([],[])
    def wave_trail(pos, t, kfreq=16, amp=0.09, L=1.2, along_dir=None):
        if along_dir is None: along_dir = np.array([1.0,0.0])
        along_dir = along_dir/np.linalg.norm(along_dir)
        perp = np.array([-along_dir[1], along_dir[0]])
        s = np.linspace(-L, 0.0, 150)
        base = pos + np.outer(s, along_dir)
        wig  = amp*np.sin(kfreq*s + 10*t)
        pts  = base + np.outer(wig, perp)
        return pts[:,0], pts[:,1]

    pos_in_start = np.array([X0, 0.0])

    def init():
        clear_point(ph_in); trail_in.set_data([],[])
        clear_point(ph_out); trail_out.set_data([],[])
        clear_point(e_vib)
        info.set_text("")
        return ph_in, trail_in, ph_out, trail_out, e_vib, info

    def animate(k):
        t = k / fps
        if t <= T_HIT:
            pos = pos_in_start + u_in * v_vis_ph_in * t
            set_point(ph_in, pos[0], pos[1])
            xs, ys = wave_trail(pos, t, along_dir=u_in); trail_in.set_data(xs, ys)
            clear_point(ph_out); trail_out.set_data([],[])
            clear_point(e_vib)
        else:
            clear_point(ph_in); trail_in.set_data([],[])
            dt = t - T_HIT
            pos_ph = u_out*(0.06 + v_vis_ph_out*dt)
            set_point(ph_out, pos_ph[0], pos_ph[1])
            xs2, ys2 = wave_trail(pos_ph, t, along_dir=u_out); trail_out.set_data(xs2, ys2)
            # Vibración pequeñísima (el electrón “no se va”)
            A = ELECTRON_WIGGLE*np.exp(-DAMPING*dt)
            vib = A*np.sin(2*np.pi*WIGGLE_FREQ*dt)
            set_point(e_vib, vib, vib)

        lines = [
            "Thomson (elástica): Eγ' = Eγ (sin cambio de energía)",
            f"Eγ = {E_gamma_keV:.2f} keV,  λ = {lam0*1e12:.3f} pm",
            f"θ (IA) ≈ {np.rad2deg(theta):.1f}°  (patrón ∝ sin²θ)",
            f"λ_C (ref) ≈ {lambda_C*1e12:.3f} pm",
        ]
        info.set_text("\n".join(lines))
        return ph_in, trail_in, ph_out, trail_out, e_vib, info

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=FRAMES, interval=1000/fps, blit=True)

    if save:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg no disponible.")
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2000,
            extra_args=["-pix_fmt","yuv420p","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-movflags","+faststart"]
        )
        out = out_path or "thomson.mp4"
        anim.save(out, writer=writer, dpi=FIG_DPI)
        print(f"Guardado: {out}")
    if show:
        plt.show()
    plt.close(fig)

# ============================================================
#                         MAIN
# ============================================================
if __name__ == "__main__":
    mode = args.mode
    fps  = args.fps

    if mode == "foto":
        anim_fotoelectric(fps=fps, save=args.save, show=args.show, out_path=args.out or "efecto_fotoelectrico.mp4")

    elif mode == "compton":
        anim_compton(fps=fps, save=args.save, show=args.show, out_path=args.out or "compton.mp4")

    elif mode == "thomson":
        anim_thomson(fps=fps, save=args.save, show=args.show, out_path=args.out or "thomson.mp4")

    elif mode == "all":
        # Ejecuta las tres; si --save, guarda tres mp4; si --show, las muestra una por una.
        anim_fotoelectric(fps=fps, save=args.save, show=args.show, out_path="efecto_fotoelectrico.mp4")
        anim_compton     (fps=fps, save=args.save, show=args.show, out_path="compton.mp4")
        anim_thomson     (fps=fps, save=args.save, show=args.show, out_path="thomson.mp4")
