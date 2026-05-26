"""
CQ-DSM Configuration v11
Speed optimisations vs v10:
  SPEED: H_MPC 24→8  (MPC re-solve period; true MPC recedes every H_MPC slots)
  SPEED: HEMS_TIME_LIMIT 1.2→0.8 s  (sub-problems are small; 0.8 s is sufficient)
  SPEED: DA_TIME_LIMIT 5.0→3.0 s  (day-ahead appliance MILP; still generous)
  SPEED: MPC_TIME_LIMIT 2.0 s  (MPC sub-problem can be slightly larger)
  SPEED: CBC cuts=off set via model attribute (no str_param gymnastics)
  FIX:   All v10 calibrations preserved (T_LOW/T_HIGH/FALLBACK_MARGIN etc.)
"""
import numpy as np

# ── Time ──────────────────────────────────────────────────────────────────────
T        = 288          # slots per day (5-min resolution)
DT       = 5 / 60      # [h] slot duration
T_AGG_S  = 3           # aggregator update period [slots]
H_HEMS   = 12          # HEMS rolling horizon [slots] = 60 min
H_MPC    = 8           # MPC receding-horizon period [slots]
                        # (was 24; MPC re-solves every H_MPC slots, not every slot)

# ── Solver time limits ────────────────────────────────────────────────────────
HEMS_TIME_LIMIT = 0.8   # [s] per rolling HEMS sub-problem (was 1.2)
DA_TIME_LIMIT   = 3.0   # [s] day-ahead appliance MILP (was 5.0)
MPC_TIME_LIMIT  = 2.0   # [s] per MPC sub-problem (larger horizon, needs more time)

# ── Prosumers ─────────────────────────────────────────────────────────────────
P           = 50
PV_FRAC     = 0.50
PV_PEAK     = 3.0      # [kW]
PV_NOISE_STD = 0.05

# ── Appliances ────────────────────────────────────────────────────────────────
E_HVAC_LO, E_HVAC_HI = 2.0, 3.5
E_DW = 1.2
E_WM = 0.8
DUR_DW = 6
DUR_WM = 8
DW_WIN  = (36,  180)
WM_WIN  = (60,  216)

# ── Base load ─────────────────────────────────────────────────────────────────
BASE_MU  = 1.5; BASE_STD = 0.3

# ── Thermal comfort ───────────────────────────────────────────────────────────
ALPHA_LO, ALPHA_HI = 0.05, 0.15
BETA_LO,  BETA_HI  = 0.8,  1.2
T_LOW  = 21.0
T_HIGH = 25.0
T_INIT = 23.0
FALLBACK_MARGIN = 0.15

# ── Battery storage ───────────────────────────────────────────────────────────
BAT_FRAC     = 0.40
BAT_CAP      = 10.0
BAT_P_MAX    = 3.0
BAT_ETA_CH   = 0.95
BAT_ETA_DIS  = 0.95
BAT_SOC_MIN  = 0.10
BAT_SOC_MAX  = 0.90
BAT_SOC_INIT = 0.50
BAT_COST_DEG = 0.002

# ── Imbalance penalty ─────────────────────────────────────────────────────────
K_IMBALANCE  = 1.5

# ── Objective weights ─────────────────────────────────────────────────────────
W1 = 1.0; W2 = 0.8; W3 = 0.5; W4 = 0.6; W5 = 0.4
MU_P = 0.5; NU_P = 0.1

# ── Adaptive weight parameters ────────────────────────────────────────────────
GAMMA_CYBER = 0.5
GAMMA_QOS   = 0.3

# ── QoS cost coefficients ─────────────────────────────────────────────────────
KAPPA_DELTA = 0.001
KAPPA_PI    = 0.05

# ── Nominal QoS ──────────────────────────────────────────────────────────────
LAT_NOM  = 0.10
PI_NOM   = 0.01

# ── Degraded QoS ─────────────────────────────────────────────────────────────
LAT_MU_DEG  = 5.0
LAT_STD_DEG = 1.5
P01 = 0.04
P10 = 0.25

# ── Communication thresholds ──────────────────────────────────────────────────
PI_MAX  = 0.25
TAU_MAX = 2 * LAT_MU_DEG   # = 10.0 s

# ── Attack parameters ─────────────────────────────────────────────────────────
FDI_START, FDI_END = 96,  192
FDI_MAG            = 0.20
FDI_MAG_STEALTHY   = 0.06
PMA_START, PMA_END = 144, 240
PMA_UP_MAX = 0.50
PMA_DN_MAX = 0.30
P_ATK_HIGH = 0.85
P_ATK_LOW  = 0.05

# ── Detector parameters ───────────────────────────────────────────────────────
DET_WINDOW = 12
DET_HIDDEN = 128
DET_THRESH = 0.50
LSTM_WINDOW = DET_WINDOW
LSTM_HIDDEN = DET_HIDDEN
LSTM_LAYERS = 2
LSTM_THRESH = DET_THRESH

# ── Multi-day evaluation ──────────────────────────────────────────────────────
NYISO_CSV = "nyiso_20230815_nyc_lbmp.csv"
DAY_SCALES = {
    "peak_summer":   1.00,
    "high_volatile": 1.18,
    "moderate":      0.82,
    "low_price":     0.61,
    "weekend":       0.75,
    "winter_peak":   1.22,
    "shoulder":      0.70,
}

# ── CBC solver options ────────────────────────────────────────────────────────
# Applied via model.cuts = 0 (not str_param) — avoids hash assertion crash
# and reduces cut-generation overhead for small sub-problems.
CBC_CUTS_OFF = True   # set model.cuts = 0 in _new_model()

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
