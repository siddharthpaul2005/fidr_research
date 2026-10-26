/* Onboard FDIR for a CubeSat-class RF gridded ion thruster - reference C.
 * Detection: CUSUM (Q16.16), steady-state Kalman bias/drift (Q8.24),
 * truncated BOCPD (float32), Poisson CUSUM on recycles.
 * Isolation: int8 linear classifier.  Recovery: rule table.            */
#include <math.h>
#include <string.h>
#include "fdir.h"
#include "fdir_params.h"

#ifndef CAL_N
#define CAL_N      1440u            /* 24 h self-calibration */
#endif
#define ISO_WIN    60
#define BO_PERSIST 15
#define VA_LIMIT   (-300.0f)

/* ---------------- on-board nominal model g(u) (float32) ---------------- */
static void nominal_model(float tp, float g[FDIR_NCH])
{
    const float dT = tp - 20.0f;
    const float mdot = MDOT_SET * (1.0f + A_T * dT);
    const float eta = IB_SET * MA_TO_UGS / mdot;
    g[0] = IB_SET;
    g[1] = IB_SET * CHI0 * (1.0f - eta) / (1.0f - ETA0);
    g[2] = IB_SET * 1e-3f * EPS_INF / (1.0f - expf(-C0M * (1.0f - eta))) / (1.0f - B_T * dT);
    g[3] = mdot;
    g[4] = VN0 + C_T * dT;
}

void fdir_init(fdir_t *f)
{
    memset(f, 0, sizeof(*f));
    for (int c = 0; c < FDIR_NCH; c++) {
        for (int r = 0; r < FDIR_R; r++) f->lp[c][r] = -INFINITY;
        f->lp[c][0] = 0.0f;
    }
    f->t_cusum = f->t_kf = f->t_bocpd = f->t_arc = f->t_alarm = -1;
    f->isolated_class = 0xFF;
}

/* ---------------- detectors on one normalised residual vector ----------- */
static inline int32_t sat32(int64_t x)
{
    return x > INT32_MAX ? INT32_MAX : (x < INT32_MIN ? INT32_MIN : (int32_t)x);
}

static void bocpd_channel(float *lp, float *s, float x, uint32_t nbase)
{
    float joint[FDIR_R];
    float mx = -INFINITY;
    for (int r = 0; r < FDIR_R - 1; r++) {           /* n == r: tables in flash */
        float m = s[r] * BO_INV_LAM[r];
        float d = x - m;
        joint[r] = lp[r] - 0.5f * d * d * BO_INV_VAR[r] - BO_HALF_LOGVAR[r];
        if (joint[r] > mx) mx = joint[r];
    }
    {   /* baseline hypothesis: n = samples since reset */
        float lam = BO_KAPPA0 + (float)nbase;
        float var = 1.0f + 1.0f / lam;
        float d = x - s[FDIR_R - 1] / lam;
        joint[FDIR_R - 1] = lp[FDIR_R - 1] - 0.5f * d * d / var - 0.5f * logf(var) - BO_HALF_L2PI;
        if (joint[FDIR_R - 1] > mx) mx = joint[FDIR_R - 1];
    }
    float acc = 0.0f;
    for (int r = 0; r < FDIR_R; r++) acc += expf(joint[r] - mx);
    const float lse = mx + logf(acc);                 /* evidence = normaliser */
    /* shift: growth probabilities and sufficient statistics */
    float hi = joint[FDIR_R - 2], lo = joint[FDIR_R - 1];
    if (lo > hi) { float t = hi; hi = lo; lo = t; }
    lp[FDIR_R - 1] = (hi == -INFINITY) ? -INFINITY : hi + log1pf(expf(lo - hi)) + BO_LOG_1MH - lse;
    s[FDIR_R - 1] += x;
    for (int r = FDIR_R - 2; r >= 1; r--) {
        lp[r] = joint[r - 1] + BO_LOG_1MH - lse;
        s[r] = s[r - 1] + x;
    }
    lp[0] = BO_LOG_H;                                 /* lse + logH - lse */
    s[0] = 0.0f;
}

void fdir_detect_z(fdir_t *f, const float z[FDIR_NCH], uint8_t n_recycle)
{
    float bo_max = 0.0f;
    int32_t cs_max = 0, kb_max = 0;
    for (int c = 0; c < FDIR_NCH; c++) {
        float zf = n_recycle ? 0.0f : z[c];           /* recycle-aware masking */
        float zr = zf * 256.0f;
        int32_t zq = (int32_t)(zr >= 0 ? zr + 0.5f : zr - 0.5f);
        if (zq > 32767) zq = 32767;
        if (zq < -32768) zq = -32768;
        /* CUSUM Q16.16 */
        int32_t z16 = zq << 8;
        int32_t sp = sat32((int64_t)f->sp[c] + z16 - CUSUM_K_Q16);
        int32_t sn = sat32((int64_t)f->sn[c] - z16 - CUSUM_K_Q16);
        f->sp[c] = sp > 0 ? sp : 0;
        f->sn[c] = sn > 0 ? sn : 0;
        int32_t cm = f->sp[c] > f->sn[c] ? f->sp[c] : f->sn[c];
        if (cm > cs_max) cs_max = cm;
        /* steady-state KF Q8.24 */
        int32_t bp = sat32((int64_t)f->kb[c] + f->kd[c]);
        int32_t nu = sat32((int64_t)(zq << 16) - bp);
        f->kb[c] = sat32((int64_t)bp + (((int64_t)nu * KF_K0_Q31 + (1LL << 30)) >> 31));
        f->kd[c] = sat32((int64_t)f->kd[c] + (((int64_t)nu * KF_K1_Q40 + (1LL << 39)) >> 40));
        int32_t ab = f->kb[c] < 0 ? -f->kb[c] : f->kb[c];
        if (ab > kb_max) kb_max = ab;
        /* BOCPD float32 */
        bocpd_channel(f->lp[c], f->s[c], zf, f->t);
        float pr = (f->t >= FDIR_R) ? -expm1f(f->lp[c][FDIR_R - 1]) : 0.0f;
        if (pr > bo_max) bo_max = pr;
        if (c == 4) { f->vn_q[f->vn_i] = (int16_t)zq; f->vn_i = (f->vn_i + 1) % FDIR_VWIN; }
    }
    /* Poisson CUSUM on recycle count */
    f->arc += (float)n_recycle * ARC_LLR_PER_EVENT - ARC_DRIFT;
    if (f->arc < 0.0f) f->arc = 0.0f;
    /* threshold logic (latched first crossings) */
    const int32_t t = (int32_t)f->t;
    f->bo_run = (bo_max > THR_BOCPD) ? (uint16_t)(f->bo_run + 1) : 0;
    if (f->t >= CAL_N) {
        if (f->t_cusum < 0 && cs_max > THR_CUSUM_Q16) f->t_cusum = t;
        if (f->t_kf < 0 && kb_max > THR_KF_Q24) f->t_kf = t;
        if (f->t_bocpd < 0 && f->bo_run >= BO_PERSIST) f->t_bocpd = t;
        if (f->t_arc < 0 && f->arc > THR_ARC) f->t_arc = t;
        int votes = (f->t_cusum >= 0) + (f->t_kf >= 0) + (f->t_bocpd >= 0);
        if (f->t_alarm < 0 && (votes >= 2 || f->t_arc >= 0)) f->t_alarm = t;
    }
    f->t++;
}

/* ---------------- isolation: int8 linear classifier -------------------- */
static uint8_t isolate(const fdir_t *f)
{
    float x[ISO_NF];
    float sabs = 0.0f;
    for (int c = 0; c < FDIR_NCH; c++) {
        x[c] = (float)f->kb[c] * (1.0f / 16777216.0f);
        sabs += fabsf(x[c]);
    }
    x[5] = log1pf(f->arc);
    float m = 0.0f, v = 0.0f;
    for (int i = 0; i < FDIR_VWIN; i++) m += f->vn_q[i];
    m /= (256.0f * FDIR_VWIN);
    for (int i = 0; i < FDIR_VWIN; i++) { float d = f->vn_q[i] / 256.0f - m; v += d * d; }
    x[6] = logf(v / FDIR_VWIN + 1e-3f);
    x[7] = log1pf(sabs);
    int32_t best = INT32_MIN; uint8_t arg = 0;
    int8_t xq[ISO_NF];
    for (int i = 0; i < ISO_NF; i++) {
        float s = (x[i] - ISO_MU[i]) * ISO_INV_SD[i] * ISO_IN_SCALE;
        s = s >= 0 ? s + 0.5f : s - 0.5f;
        xq[i] = (int8_t)(s > 127 ? 127 : (s < -127 ? -127 : (int32_t)s));
    }
    for (int k = 0; k < ISO_NC; k++) {
        int32_t acc = ISO_BQ[k];
        for (int i = 0; i < ISO_NF; i++) acc += (int32_t)xq[i] * ISO_WQ[i][k];
        if (acc > best) { best = acc; arg = (uint8_t)k; }
    }
    return arg;
}

/* ---------------- full step: model -> residual -> detect -> isolate -> recover */
fdir_action_t fdir_step(fdir_t *f, const fdir_frame_t *y, float va_cmd_V)
{
    float g[FDIR_NCH], r[FDIR_NCH], z[FDIR_NCH];
    const float yy[FDIR_NCH] = { y->ib_mA, y->ia_mA, y->prf_W, y->mdot_ugs, y->vn_V };
    nominal_model(y->tp_C, g);
    const float dT = y->tp_C - 20.0f;
    for (int c = 0; c < FDIR_NCH; c++) {
        r[c] = yy[c] - g[c];
        if (f->t < CAL_N && !y->n_recycle) {          /* accumulate centred LS sums */
            if (f->c_sw[c] == 0.0f) f->c_r0[c] = r[c];
            float rc = r[c] - f->c_r0[c];
            f->c_sw[c] += 1.0f; f->c_sx[c] += dT; f->c_sy[c] += rc;
            f->c_sxx[c] += dT * dT; f->c_sxy[c] += dT * rc; f->c_syy[c] += rc * rc;
        }
        if (f->t == CAL_N) {                           /* solve 2x2 normal equations */
            float det = f->c_sw[c] * f->c_sxx[c] - f->c_sx[c] * f->c_sx[c];
            float b = (f->c_sw[c] * f->c_sxy[c] - f->c_sx[c] * f->c_sy[c]) / det;
            float m = (f->c_sy[c] - b * f->c_sx[c]) / f->c_sw[c];
            float sse = f->c_syy[c] - m * f->c_sy[c] - b * f->c_sxy[c];
            f->beta[c] = b; f->mu[c] = m + f->c_r0[c];
            f->inv_sd[c] = 1.0f / sqrtf(sse / f->c_sw[c]);
        }
        z[c] = (f->t >= CAL_N) ? (r[c] - f->mu[c] - f->beta[c] * dT) * f->inv_sd[c] : 0.0f;
    }
    fdir_detect_z(f, z, y->n_recycle);

    if (f->t_alarm >= 0 && f->isolated_class == 0xFF &&
        (int32_t)f->t - 1 >= f->t_alarm + ISO_WIN) {
        f->isolated_class = isolate(f);
        fdir_action_t a = (fdir_action_t)(f->isolated_class + 1);   /* rule table */
        if (a == ACT_VA_CORRECT && va_cmd_V <= VA_LIMIT) a = ACT_THROTTLE_POWER;
        f->action = (uint8_t)a;
        return a;
    }
    return ACT_NONE;
}

float fdir_dbg_cusum(const fdir_t *f)
{
    int32_t m = 0;
    for (int c = 0; c < FDIR_NCH; c++) {
        if (f->sp[c] > m) m = f->sp[c];
        if (f->sn[c] > m) m = f->sn[c];
    }
    return m / 65536.0f;
}
float fdir_dbg_kf(const fdir_t *f)
{
    int32_t m = 0;
    for (int c = 0; c < FDIR_NCH; c++) { int32_t a = f->kb[c] < 0 ? -f->kb[c] : f->kb[c]; if (a > m) m = a; }
    return m / 16777216.0f;
}
float fdir_dbg_bocpd(const fdir_t *f)
{
    float m = 0;
    for (int c = 0; c < FDIR_NCH; c++) { float p = -expm1f(f->lp[c][FDIR_R - 1]); if (p > m) m = p; }
    return m;
}
