/* Onboard FDIR for a CubeSat-class RF gridded ion thruster.
 * Target: ARM Cortex-M4F (single-precision FPU). One call per 60 s telemetry frame. */
#ifndef FDIR_H
#define FDIR_H
#include <stdint.h>

#define FDIR_NCH   5      /* Ib, Ia, Prf, mdot, Vn */
#define FDIR_R     64     /* BOCPD run-length hypotheses (truncated) */
#define FDIR_VWIN  61     /* Vn variance window for isolation feature */

typedef struct {          /* one 60 s telemetry frame */
    float ib_mA, ia_mA, prf_W, mdot_ugs, vn_V, tp_C;
    uint8_t n_recycle;    /* PPU recycle events in this frame */
} fdir_frame_t;

typedef enum { ACT_NONE = 0, ACT_VA_CORRECT = 1, ACT_SWITCH_CATHODE = 2,
               ACT_THROTTLE_HV_COND = 3, ACT_FEED_SETPOINT = 4,
               ACT_THROTTLE_POWER = 5 } fdir_action_t;

typedef struct {
    /* self-calibration: offset + thermal gain least squares (centred sums) */
    float c_r0[FDIR_NCH], c_sw[FDIR_NCH], c_sx[FDIR_NCH], c_sy[FDIR_NCH],
          c_sxx[FDIR_NCH], c_sxy[FDIR_NCH], c_syy[FDIR_NCH];
    float mu[FDIR_NCH], beta[FDIR_NCH], inv_sd[FDIR_NCH];
    /* CUSUM, Q16.16 */
    int32_t sp[FDIR_NCH], sn[FDIR_NCH];
    /* steady-state Kalman bias/drift, Q8.24 */
    int32_t kb[FDIR_NCH], kd[FDIR_NCH];
    /* truncated BOCPD (float32); n[r] == r for r < R-1, so only sums are stored */
    float lp[FDIR_NCH][FDIR_R], s[FDIR_NCH][FDIR_R];
    /* Poisson CUSUM on recycle counts */
    float arc;
    /* isolation buffer for Vn */
    int16_t vn_q[FDIR_VWIN];
    uint8_t vn_i;
    /* bookkeeping */
    uint32_t t;
    uint16_t bo_run;
    int32_t t_cusum, t_kf, t_bocpd, t_arc, t_alarm;
    uint8_t isolated_class;   /* 0..4, 0xFF = none */
    uint8_t action;
} fdir_t;

void fdir_init(fdir_t *f);
/* returns action to execute this frame (ACT_NONE most of the time) */
fdir_action_t fdir_step(fdir_t *f, const fdir_frame_t *y, float va_cmd_V);

/* debug taps for equivalence testing */
float fdir_dbg_cusum(const fdir_t *f);
float fdir_dbg_kf(const fdir_t *f);
float fdir_dbg_bocpd(const fdir_t *f);
#endif
