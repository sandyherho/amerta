"""
Saint-Venant 1D visualization — v0.0.4.
Dark-themed figures + GIF animation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
from tqdm import tqdm

BG     = '#0d1b2a'
PANEL  = '#1b263b'
EDGE   = '#415a77'
MUTED  = '#778da9'
LIGHT  = '#e0e1dd'
ACCENT = '#e0a458'
WAVE   = '#5fa8d3'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.1,
})


def _style_axes(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(EDGE)
    ax.tick_params(colors=LIGHT, which='both')
    ax.xaxis.label.set_color(LIGHT); ax.yaxis.label.set_color(LIGHT)
    ax.title.set_color(LIGHT)
    ax.grid(True, color=EDGE, alpha=0.3, linewidth=0.5)


class Animator:
    @staticmethod
    def fig_time_evolution(result, filename, output_dir, scenario_name, dpi=150):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        x = result['x']
        n_snaps = 5
        all_idx = np.round(np.linspace(0, len(result['t_all'])-1, n_snaps)).astype(int)
        t = result['t_all'][all_idx]
        h = result['h_all'][all_idx]
        u = result['u_all'][all_idx]
        q = result['q_all'][all_idx]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor=BG, sharex=True)
        cmap = plt.get_cmap('viridis')
        for i in range(len(t)):
            c = cmap(i/max(len(t)-1,1))
            axes[0].plot(x, h[i], color=c, lw=1.6, label=f't={t[i]:.2f}s')
            axes[1].plot(x, u[i], color=c, lw=1.6)
            axes[2].plot(x, q[i], color=c, lw=1.6)
        axes[0].set_ylabel('h [m]'); axes[0].set_title(f'{scenario_name}: Time Evolution', fontweight='bold')
        axes[1].set_ylabel('u [m/s]'); axes[2].set_ylabel('q [m$^2$/s]')
        axes[2].set_xlabel('x [m]')
        axes[0].legend(loc='upper right', facecolor=PANEL, edgecolor=EDGE, labelcolor=LIGHT, fontsize=8)
        for ax in axes: _style_axes(ax)
        plt.tight_layout()
        plt.savefig(fp, dpi=dpi, facecolor=BG); plt.close(fig)
        return str(fp)

    @staticmethod
    def fig_physical(result, filename, output_dir, scenario_name, dpi=150):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        x = result['x']; h = result['h_final']; q = result['q_final']
        g = result['params']['g']
        u = np.where(h>1e-8, q/h, 0.0); Fr = np.abs(u)/np.sqrt(g*h+1e-12)
        E = 0.5*u*u + g*h
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=BG)
        axes[0,0].plot(x, h, color=WAVE, lw=1.8); axes[0,0].fill_between(x, 0, h, color=WAVE, alpha=0.25)
        axes[0,0].set_ylabel('h [m]'); axes[0,0].set_title('Final depth profile', color=LIGHT)
        axes[0,1].plot(x, u, color=ACCENT, lw=1.8); axes[0,1].set_ylabel('u [m/s]')
        axes[0,1].set_title('Final velocity', color=LIGHT)
        axes[1,0].plot(x, Fr, color='#ef6f6c', lw=1.8)
        axes[1,0].axhline(1.0, color=MUTED, ls='--', lw=1)
        axes[1,0].set_ylabel('Fr'); axes[1,0].set_xlabel('x [m]')
        axes[1,0].set_title('Froude number', color=LIGHT)
        axes[1,1].plot(x, E, color='#9dd1c7', lw=1.8); axes[1,1].set_ylabel('E [m]')
        axes[1,1].set_xlabel('x [m]'); axes[1,1].set_title('Specific energy head', color=LIGHT)
        for ax in axes.flat: _style_axes(ax)
        fig.suptitle(f'{scenario_name}: Physical Interpretation', color=LIGHT, fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(fp, dpi=dpi, facecolor=BG); plt.close(fig)
        return str(fp)

    @staticmethod
    def fig_numerical(result, filename, output_dir, scenario_name, dpi=150):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        p = result['params']
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=BG)
        x = result['x']; h = result['h_final']
        c = np.sqrt(p['g']*h); axes[0,0].plot(x, c, color=WAVE, lw=1.6)
        axes[0,0].set_ylabel('c = sqrt(gh) [m/s]'); axes[0,0].set_xlabel('x [m]')
        axes[0,0].set_title('Wave celerity (final)', color=LIGHT)
        labels = ['dt_min','dt_max']; vals = [result['dt_min'], result['dt_max']]
        axes[0,1].bar(labels, vals, color=[WAVE, ACCENT], edgecolor=EDGE)
        axes[0,1].set_ylabel('dt [s]'); axes[0,1].set_title('Adaptive time step range', color=LIGHT)
        axes[1,0].bar(['target','mean','max'],
                      [p['cfl'], result.get('cfl_mean', result['cfl_max']), result['cfl_max']],
                      color=[MUTED, WAVE, ACCENT], edgecolor=EDGE)
        axes[1,0].axhline(1.0, color='#ef6f6c', ls='--', lw=1)
        axes[1,0].set_ylabel('CFL'); axes[1,0].set_title('CFL statistics', color=LIGHT)
        axes[1,1].bar(['final','max |err|'],
                      [abs(result['mass_err_pct']), result['max_mass_err']],
                      color=[WAVE, ACCENT], edgecolor=EDGE)
        axes[1,1].set_ylabel('|mass error| [%]')
        axes[1,1].set_title('Conservation diagnostics', color=LIGHT)
        if abs(result['mass_err_pct']) > 0 or result['max_mass_err'] > 0:
            axes[1,1].set_yscale('log')
        for ax in axes.flat: _style_axes(ax)
        fig.suptitle(f'{scenario_name}: Numerical Aspects', color=LIGHT, fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(fp, dpi=dpi, facecolor=BG); plt.close(fig)
        return str(fp)

    @staticmethod
    def create_gif(result, filename, output_dir, scenario_name, fps=15, dpi=110):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        x = result['x']; H = result['anim_h']; U = result['anim_u']; T = result['anim_times']
        p = result['params']
        L = float(p['L']); x_dam = 0.5 * L
        nF = H.shape[0]
        fig = plt.figure(figsize=(9, 6.8), facecolor=BG)
        gs = fig.add_gridspec(2, 1, left=0.10, right=0.97,
                              top=0.83, bottom=0.09, hspace=0.18)
        axh = fig.add_subplot(gs[0, 0]); axu = fig.add_subplot(gs[1, 0], sharex=axh)
        _style_axes(axh); _style_axes(axu)
        axh.set_ylabel('h [m]'); axu.set_ylabel('u [m/s]'); axu.set_xlabel('x [m]')
        axh.set_xlim(x.min(), x.max())
        axh.set_ylim(0, float(np.max(H))*1.08)
        u_abs = float(np.max(np.abs(U))) + 1e-6
        axu.set_ylim(-u_abs*1.1, u_abs*1.1)
        plt.setp(axh.get_xticklabels(), visible=False)

        hL_p = float(p['h_left']); hR_p = float(p['h_right'])
        uL_p = float(p.get('u_left', 0.0)); uR_p = float(p.get('u_right', 0.0))
        g_p  = float(p['g'])
        subtitle = (f"idealized dam break  |  h$_L$={hL_p:g} m, h$_R$={hR_p:g} m,  "
                    f"u$_L$={uL_p:+g} m/s, u$_R$={uR_p:+g} m/s  |  "
                    f"L={L:g} m, g={g_p:g} m/s$^2$")

        fig.text(0.5, 0.962, scenario_name, ha='center', va='center',
                 color=LIGHT, fontweight='bold', fontsize=13)
        fig.text(0.5, 0.925, subtitle, ha='center', va='center',
                 color=MUTED, fontsize=9)
        ttxt = fig.text(0.5, 0.888, '', ha='center', va='center',
                        color=ACCENT, fontfamily='monospace', fontsize=11)

        REF = '#ef4444'
        axh.axvline(x_dam, color=REF, ls='--', lw=1.3, alpha=0.85, zorder=2)
        axu.axvline(x_dam, color=REF, ls='--', lw=1.3, alpha=0.85, zorder=2)

        (fill,) = axh.fill(np.r_[x, x[::-1]], np.r_[H[0], np.zeros_like(x)[::-1]],
                           color=WAVE, alpha=0.35, zorder=3)
        (lineh,) = axh.plot(x, H[0], color=WAVE, lw=1.8, zorder=4)
        (lineu,) = axu.plot(x, U[0], color=ACCENT, lw=1.6, zorder=4)

        def update(i):
            lineh.set_ydata(H[i]); lineu.set_ydata(U[i])
            xy = np.column_stack([np.r_[x, x[::-1]], np.r_[H[i], np.zeros_like(x)[::-1]]])
            fill.set_xy(xy)
            ttxt.set_text(f't = {T[i]:6.3f} s')
            return lineh, lineu, fill, ttxt

        anim = animation.FuncAnimation(fig, update, frames=nF, interval=1000/fps, blit=False)
        writer = animation.PillowWriter(fps=fps)
        with tqdm(total=nF, desc="  Rendering GIF", unit="frame") as pbar:
            def cb(cur, tot): pbar.n = cur+1; pbar.refresh()
            anim.save(fp, writer=writer, dpi=dpi,
                      savefig_kwargs={'facecolor': BG},
                      progress_callback=cb)
        plt.close(fig)
        return str(fp)
