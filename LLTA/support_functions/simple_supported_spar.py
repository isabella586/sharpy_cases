"""
Spar Vibration Mode Calculator - Simply Supported (Pinned-Pinned) Beam
=======================================================================

GOVERNING EQUATION (Euler-Bernoulli Beam Theory)
-------------------------------------------------
For a simply supported beam the natural frequencies are:

    f_n = (n^2 * pi^2) / (2*pi * L^2) * sqrt(EI / mass_per_length)   [Hz]
    omega_n = (n*pi/L)^2 * sqrt(EI / mass_per_length)                [rad/s]

where:
    n        = mode number (1, 2, 3, ...)
    L        = span length                         [m]
    E        = Young modulus                       [Pa]
    I        = second moment of area               [m^4]
    EI       = flexural rigidity                   [N.m^2]
    mass_per_length      = mass per unit length                [kg/m]

The mode shapes (normalised transverse displacement) are:
    phi_n(x) = sin(n*pi*x / L)

MASS PER UNIT LENGTH
--------------------
mass_per_length can be supplied in two ways:
  - Analytically:  mass_per_length_analytical = rho * A   (computed from density and area)
  - Measured:      mass_per_length_measured -> supplied directly (e.g. from weighing the spar)

A measured mass_per_length accounts for resin-rich regions, ply drops, adhesive joints,
surface coatings, and other features not captured by the nominal geometry.
The reported percentage difference is:

    Delta_mass_per_length (%) = 100 * (mass_per_length_measured - mass_per_length_analytical) / mass_per_length_analytical

MEASURED EI
-----------
EI can also be supplied as a measured value (e.g. from a three-point bend test)
in place of the analytical E*I.  The reported percentage difference is:

    Delta_f_n (%) = 100 * (f_n_meas - f_n_anal) / f_n_anal

The four combinations of (EI, mass_per_length) that can be compared are:
    [1] Analytical EI,  Analytical mass_per_length  -- fully nominal
    [2] Measured   EI,  Analytical mass_per_length  -- stiffness effect only
    [3] Analytical EI,  Measured   mass_per_length  -- mass effect only
    [4] Measured   EI,  Measured   mass_per_length  -- fully measured

ASSUMPTIONS
-----------
1.  Euler-Bernoulli beam theory: plane sections remain plane; shear
    deformation and rotary inertia are negligible (valid for L/r >= 10).
2.  Prismatic beam: uniform properties along the full span.
3.  Linearly elastic, homogeneous material.
4.  Small deflections (linear geometry).
5.  Ideal simply supported BCs: zero displacement and zero bending moment
    at both ends; ends free to rotate.
6.  No axial pre-stress, no damping, no elastic foundation.
7.  Measured EI and mass_per_length each represent uniform effective values over the
    full span.  Spatially varying properties require a numerical solution.
"""

import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SparProperties:
    """
    Properties of a simply supported spar.

    Required inputs
    ---------------
    L   : span length [m]
    E   : Young modulus [Pa]
    I   : second moment of area [m^4]
    rho : material density [kg/m^3]
    A   : cross-sectional area [m^2]

    Optional measured overrides
    ---------------------------
    EI_measured  : measured flexural rigidity [N.m^2]
    mass_per_length_measured : measured mass per unit length [kg/m]
    """
    L:   float
    E:   float
    I:   float
    rho: float
    A:   float
    name: str = "Spar"
    EI_measured:  Optional[float] = None
    mass_per_length_measured: Optional[float] = None

    # -- Analytical values --------------------------------------------------

    @property
    def EI_analytical(self) -> float:
        """E * I  [N.m^2]"""
        return self.E * self.I

    @property
    def mass_per_length_analytical(self) -> float:
        """rho * A  [kg/m]"""
        return self.rho * self.A

    @property
    def EI(self) -> float:
        return self.EI_analytical

    # -- Ratios (None when measurement not supplied) -------------------------

    @property
    def EI_ratio(self) -> Optional[float]:
        if self.EI_measured is None:
            return None
        return self.EI_measured / self.EI_analytical

    @property
    def mass_per_length_ratio(self) -> Optional[float]:
        if self.mass_per_length_measured is None:
            return None
        return self.mass_per_length_measured / self.mass_per_length_analytical

    # -- Slenderness (uses analytical I, A) ----------------------------------

    @property
    def slenderness(self) -> float:
        r = math.sqrt(self.I / self.A)
        return self.L / r


@dataclass
class VibrationMode:
    n:          int
    omega:      float    # [rad/s]
    freq_hz:    float    # [Hz]
    wavenumber: float    # k_n = n*pi/L  [1/m]
    ei_source:  str = "analytical"   # "analytical" | "measured"
    mass_per_length_source: str = "analytical"   # "analytical" | "measured"


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _freq_from_EI_mass_per_length(n, L, EI, mass_per_length, ei_src, mass_per_length_src):
    k_n     = n * math.pi / L
    omega_n = k_n**2 * math.sqrt(EI / mass_per_length)
    return VibrationMode(
        n=n, omega=omega_n,
        freq_hz=omega_n / (2.0 * math.pi),
        wavenumber=k_n,
        ei_source=ei_src, mass_per_length_source=mass_per_length_src,
    )


def _resolve(spar, use_meas_EI, use_meas_mass_per_length):
    """Return the (EI, mass_per_length, ei_src, mass_per_length_src) tuple to use."""
    if use_meas_EI:
        if spar.EI_measured is None:
            raise ValueError("EI_measured is not set.")
        EI     = spar.EI_measured
        ei_src = "measured"
    else:
        EI     = spar.EI_analytical
        ei_src = "analytical"

    if use_meas_mass_per_length:
        if spar.mass_per_length_measured is None:
            raise ValueError("mass_per_length_measured is not set.")
        mass_per_length     = spar.mass_per_length_measured
        mass_per_length_src = "measured"
    else:
        mass_per_length     = spar.mass_per_length_analytical
        mass_per_length_src = "analytical"

    return EI, mass_per_length, ei_src, mass_per_length_src


def compute_modes(spar, n_modes=10,
                  use_meas_EI=False, use_meas_mass_per_length=False):
    """
    Return the first n_modes VibrationMode objects.

    Parameters
    ----------
    use_meas_EI  : if True, use spar.EI_measured  instead of E*I
    use_meas_mass_per_length : if True, use spar.mass_per_length_measured instead of rho*A
    """
    EI, mass_per_length, ei_src, mass_per_length_src = _resolve(spar, use_meas_EI, use_meas_mass_per_length)
    return [_freq_from_EI_mass_per_length(n, spar.L, EI, mass_per_length, ei_src, mass_per_length_src)
            for n in range(1, n_modes + 1)]


# ---------------------------------------------------------------------------
# Mode shape
# ---------------------------------------------------------------------------

def mode_shape(n, spar, x_points=200):
    """Normalised mode shape phi_n(x) = sin(n*pi*x/L).  Returns (x, phi)."""
    x   = np.linspace(0, spar.L, x_points)
    phi = np.sin(n * np.pi * x / spar.L)
    return x, phi


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

HDR = ("  %5s  %12s  %14s  %12s  %10s"
       % ("Mode", "k_n [1/m]", "omega_n [rad/s]", "f_n [Hz]", "T_n [ms]"))
DIVIDER = "  " + "-" * 60


def _row(m):
    T_ms = 1000.0 / m.freq_hz
    return ("  %5d  %12.4f  %14.4f  %12.4f  %10.3f"
            % (m.n, m.wavenumber, m.omega, m.freq_hz, T_ms))


def _print_modes(modes, label):
    print()
    print("  --- %s ---" % label)
    print()
    print(HDR)
    print(DIVIDER)
    for m in modes:
        print(_row(m))


def _print_comparison(label_a, label_b, modes_a, modes_b):
    """Two-column frequency delta table."""
    print()
    print("  --- Comparison: %s vs %s ---" % (label_b, label_a))
    print()
    print("  %5s  %14s  %14s  %13s  %11s"
          % ("Mode", label_a[:14], label_b[:14], "Delta f [Hz]", "Delta f [%]"))
    print(DIVIDER)
    for ma, mb in zip(modes_a, modes_b):
        df    = mb.freq_hz - ma.freq_hz
        df_pc = 100.0 * df / ma.freq_hz
        print("  %5d  %14.4f  %14.4f  %+13.4f  %+11.2f%%"
              % (ma.n, ma.freq_hz, mb.freq_hz, df, df_pc))


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def print_report(spar, n_modes=6):
    sep = "=" * 72

    print(sep)
    print("  SPAR VIBRATION ANALYSIS - Simply Supported (Pinned-Pinned)")
    print(sep)
    print()
    print("  Spar : %s" % spar.name)
    print("  %-36s %.4f m"      % ("Span length L",                  spar.L))
    print("  %-36s %.4e Pa"     % ("Young modulus E",                 spar.E))
    print("  %-36s %.4e m^4"    % ("Second moment of area I",         spar.I))
    print("  %-36s %.2f kg/m^3" % ("Density rho",                     spar.rho))
    print("  %-36s %.4e m^2"    % ("Cross-sectional area A",          spar.A))
    print()

    # EI block
    print("  %-36s %.4e N.m^2"  % ("EI analytical (E*I)",             spar.EI_analytical))
    if spar.EI_measured is not None:
        ei_pct = 100.0 * (spar.EI_ratio - 1.0)
        print("  %-36s %.4e N.m^2"      % ("EI measured",             spar.EI_measured))
        print("  %-36s %.4f  (%+.2f%%)" % ("EI ratio (meas/anal)",    spar.EI_ratio, ei_pct))
        E_eff = spar.EI_measured / spar.I
        print("  %-36s %.4e Pa  (delta %+.2f%%)"
              % ("Back-calc E_eff (EI_meas/I)", E_eff, 100.0*(E_eff/spar.E - 1.0)))
    print()

    # mass_per_length block
    print("  %-36s %.4f kg/m"   % ("mass_per_length analytical (rho*A)",          spar.mass_per_length_analytical))
    if spar.mass_per_length_measured is not None:
        mass_per_length_pct = 100.0 * (spar.mass_per_length_ratio - 1.0)
        print("  %-36s %.4f kg/m"         % ("mass_per_length measured",          spar.mass_per_length_measured))
        print("  %-36s %.4f  (%+.2f%%)"   % ("mass_per_length ratio (meas/anal)", spar.mass_per_length_ratio, mass_per_length_pct))
    print()

    print("  %-36s %.1f"        % ("Slenderness ratio L/r",            spar.slenderness))
    if spar.slenderness < 10:
        print()
        print("  WARNING: Slenderness < 10.  Shear deformation may be significant.")
        print("  Consider Timoshenko beam theory.")

    has_EI  = spar.EI_measured  is not None
    has_mass_per_length = spar.mass_per_length_measured is not None

    # Always compute case [1]: fully analytical
    m_aa = compute_modes(spar, n_modes, use_meas_EI=False, use_meas_mass_per_length=False)
    _print_modes(m_aa, "Case [1] Analytical EI,  Analytical mass_per_length  (fully nominal)")

    print()
    print("  Frequency ratio check (f_n/f_1 should equal n^2):")
    f1 = m_aa[0].freq_hz
    for m in m_aa:
        print("    Mode %d: %.4f  (expected %d)" % (m.n, m.freq_hz/f1, m.n**2))

    if has_EI:
        m_ma = compute_modes(spar, n_modes, use_meas_EI=True,  use_meas_mass_per_length=False)
        _print_modes(m_ma, "Case [2] Measured EI,    Analytical mass_per_length  (stiffness effect only)")
        _print_comparison("anal EI / anal mass_per_length", "meas EI / anal mass_per_length", m_aa, m_ma)

    if has_mass_per_length:
        m_am = compute_modes(spar, n_modes, use_meas_EI=False, use_meas_mass_per_length=True)
        _print_modes(m_am, "Case [3] Analytical EI,  Measured mass_per_length    (mass effect only)")
        _print_comparison("anal EI / anal mass_per_length", "anal EI / meas mass_per_length", m_aa, m_am)

    if has_EI and has_mass_per_length:
        m_mm = compute_modes(spar, n_modes, use_meas_EI=True,  use_meas_mass_per_length=True)
        _print_modes(m_mm, "Case [4] Measured EI,    Measured mass_per_length    (fully measured)")
        _print_comparison("anal EI / anal mass_per_length", "meas EI / meas mass_per_length", m_aa, m_mm)

    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Section geometry helpers
# ---------------------------------------------------------------------------

def hollow_circular_I(D_outer, D_inner):
    """Second moment of area for a hollow circular cross-section [m^4]."""
    return math.pi / 64.0 * (D_outer**4 - D_inner**4)


def hollow_circular_A(D_outer, D_inner):
    """Cross-sectional area for a hollow circular cross-section [m^2]."""
    return math.pi / 4.0 * (D_outer**2 - D_inner**2)


# ---------------------------------------------------------------------------
# Examass_per_lengthe: CFRP hollow circular spar
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Geometry
    D_o = 0.01414    # [NEED UPDATING FROM VOLTITUDE DATA] Outer diameter [m]
    D_i = 0.01176    # [NEED UPDATING FROM VOLTITUDE DATA] Inner diameter [m]  (4 mm wall thickness)
    L   = 1        # Span [m]

    I_cf = hollow_circular_I(D_o, D_i)
    A_cf = hollow_circular_A(D_o, D_i)

    # Nominal CFRP material properties (unidirectional, 0-deg dominated layup)
    E_cf = 135e9   # Longitudinal Young modulus [Pa]
    rho_cf = 1550.0  # Density [kg/m^3]

    # Analytical values for reference
    EI_anal  = E_cf * I_cf
    mass_per_length_anal = rho_cf * A_cf

    # Simulated measurements
    # EI_measured: 3-point bend test gave a value 6% below nominal
    #   (e.g. slightly lower fibre volume fraction than assumed)
    # mass_per_length_measured: direct weighing gave a value 4% above nominal
    #   (e.g. extra resin, adhesive at joints)
    EI_measured  = 51.25 # Nm^2, updated to EI from Voltitude
    mass_per_length_measured = 2.83 / 9.81 # kg/m, updated to mass_per_length from Voltitude

    cfrp_spar = SparProperties(
        L            = L,
        E            = E_cf,
        I            = I_cf,
        rho          = rho_cf,
        A            = A_cf,
        name         = "CFRP hollow circular spar",
        EI_measured  = EI_measured,
        mass_per_length_measured = mass_per_length_measured,
    )

    print_report(cfrp_spar, n_modes=6)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        spar    = cfrp_spar
        n_modes = 6

        m_aa = compute_modes(spar, n_modes, False, False)
        m_ma = compute_modes(spar, n_modes, True,  False)
        m_am = compute_modes(spar, n_modes, False, True)
        m_mm = compute_modes(spar, n_modes, True,  True)

        cases = {
            "[1] Anal EI / Anal mass_per_length": (m_aa, "steelblue"),
            "[2] Meas EI / Anal mass_per_length": (m_ma, "darkorange"),
            "[3] Anal EI / Meas mass_per_length": (m_am, "seagreen"),
            "[4] Meas EI / Meas mass_per_length": (m_mm, "crimson"),
        }

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(spar.name, fontsize=11, fontweight="bold")
        gs  = gridspec.GridSpec(2, n_modes//2 + 1,
                                width_ratios=[1]*(n_modes//2) + [1.6])

        # Mode shape panels (shape is identical for all cases; annotate freqs)
        for idx in range(n_modes):
            row = idx // (n_modes // 2)
            col = idx %  (n_modes // 2)
            ax  = fig.add_subplot(gs[row, col])
            x, phi = mode_shape(idx+1, spar)
            ax.plot(x, phi, color="steelblue", lw=2)
            ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.fill_between(x, phi, alpha=0.12, color="steelblue")

            lines = ["Mode %d" % (idx+1)]
            for lbl, (modes, _) in cases.items():
                lines.append("%s: %.2f Hz" % (lbl[:3], modes[idx].freq_hz))
            ax.set_title("\n".join(lines), fontsize=7)
            ax.set_xlabel("x [m]", fontsize=7)
            ax.set_ylabel("phi(x)", fontsize=7)
            ax.set_xlim(0, spar.L)
            ax.set_ylim(-1.3, 1.3)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

        # Grouped bar chart comparing all four cases
        ax_bar   = fig.add_subplot(gs[:, -1])
        n_cases  = len(cases)
        mode_nums = [m.n for m in m_aa]
        x_pos    = np.arange(len(mode_nums), dtype=float)
        bw       = 0.18   # bar width
        offsets  = np.linspace(-(n_cases-1)/2*bw, (n_cases-1)/2*bw, n_cases)

        for (lbl, (modes, colour)), offset in zip(cases.items(), offsets):
            freqs = [m.freq_hz for m in modes]
            bars  = ax_bar.bar(x_pos + offset, freqs, bw,
                               label=lbl, color=colour, alpha=0.82)

        # Annotate % delta of case [4] vs case [1] above each group
        for i, (ma, mm) in enumerate(zip(m_aa, m_mm)):
            pc  = 100.0 * (mm.freq_hz - ma.freq_hz) / ma.freq_hz
            top = max(ma.freq_hz, mm.freq_hz) * 1.02
            ax_bar.text(i, top, "%+.1f%%" % pc,
                        ha="center", va="bottom", fontsize=6.5, color="dimgray")

        ax_bar.set_xlabel("Mode number")
        ax_bar.set_ylabel("Frequency [Hz]")
        ax_bar.set_title("Natural Frequencies\nby EI / mass_per_length combination", fontsize=9)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(["M%d" % n for n in mode_nums])
        ax_bar.legend(fontsize=6.5, loc="upper left")
        ax_bar.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        out = "/Users/icc25/sharpy_cases/LLTA/spar_comparison_cfrp.png"
        plt.savefig(out, dpi=150)
        print("  Plot saved -> %s" % out)
        plt.show()

    except ImportError:
        print("  (matplotlib not available - skipping plots)")
