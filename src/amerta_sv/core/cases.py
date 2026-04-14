"""Canonical Riemann-problem presets for 1D SWE."""
CASES = {
    "stoker": {"desc": "Stoker (1957) wet-bed dam break: rarefaction + shock",
               "h_left": 2.0, "h_right": 0.5, "u_left": 0.0, "u_right": 0.0,
               "t_final": 6.0, "L": 100.0},
    "ritter": {"desc": "Ritter (1892) dry-bed dam break: single rarefaction",
               "h_left": 1.0, "h_right": 1.0e-6, "u_left": 0.0, "u_right": 0.0,
               "t_final": 4.0, "L": 100.0},
    "double_rarefaction": {"desc": "Symmetric double rarefaction: diverging flow",
                           "h_left": 1.0, "h_right": 1.0, "u_left": -5.0, "u_right": 5.0,
                           "t_final": 2.5, "L": 100.0},
    "double_shock": {"desc": "Symmetric double shock: converging flow / shock collision",
                     "h_left": 1.0, "h_right": 1.0, "u_left": 5.0, "u_right": -5.0,
                     "t_final": 2.0, "L": 100.0},
}
def get_case(name):
    if name not in CASES:
        raise ValueError(f"Unknown case '{name}'. Available: {list(CASES.keys())}")
    return dict(CASES[name])
