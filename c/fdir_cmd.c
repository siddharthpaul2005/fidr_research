#include "fdir_cmd.h"

void cmd_init(cmd_state_t *s)
{
    s->va_V = -200; s->beam_dmA = 150; s->neutraliser = 0; s->seq = 0;
    s->last_action_min = 0; s->safe_mode = 0;
    for (int i = 0; i < 6; i++) s->n_actions[i] = 0;
}

uint16_t crc16_ccitt(const uint8_t *d, int n)
{
    uint16_t c = 0xFFFF;
    for (int i = 0; i < n; i++) {
        c ^= (uint16_t)d[i] << 8;
        for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}

static void frame(cmd_state_t *s, uint8_t f[8], uint8_t op, int16_t arg)
{
    f[0] = 0xEB; f[1] = 0x90; f[2] = s->seq++; f[3] = op;
    f[4] = (uint8_t)(arg & 0xFF); f[5] = (uint8_t)((uint16_t)arg >> 8);
    uint16_t c = crc16_ccitt(f, 6);
    f[6] = (uint8_t)(c >> 8); f[7] = (uint8_t)(c & 0xFF);
}

static int throttle(cmd_state_t *s, uint8_t f[8])
{
    int16_t b = (int16_t)(s->beam_dmA * 9 / 10);          /* -10 % */
    if (b < CMD_BEAM_MIN_dmA) return 0;
    s->beam_dmA = b;
    frame(s, f, OP_SET_BEAM, b);
    return 1;
}

int cmd_encode(cmd_state_t *s, int action, uint32_t now_min, uint8_t out[][8])
{
    if (action < 1 || action > 5 || s->safe_mode) return 0;
    if (s->last_action_min && now_min - s->last_action_min < CMD_MIN_GAP_MIN) return 0;
    if (++s->n_actions[action] > CMD_MAX_PER_CLASS) { s->safe_mode = 1; return 0; }
    int n = 0;
    switch (action) {
    case 1: {                                   /* grid erosion: Va -25 V, else throttle */
        int16_t v = (int16_t)(s->va_V - 25);
        if (v >= CMD_VA_MIN_V) { s->va_V = v; frame(s, out[n++], OP_SET_VA, v); }
        else n += throttle(s, out[n]);
        break; }
    case 2:                                      /* neutraliser wear: switch to the other unit */
        if (s->neutraliser == 1) { s->safe_mode = 1; return 0; }   /* no spare left */
        s->neutraliser = 1; frame(s, out[n++], OP_SEL_NEUTRAL, 1);
        break;
    case 3:                                      /* arcing: throttle + 3 conditioning cycles */
        n += throttle(s, out[n]);
        frame(s, out[n++], OP_HV_CONDITION, 3);
        break;
    case 4:                                      /* feed: +1.0 degC sublimator set point */
        frame(s, out[n++], OP_FEED_HEATER, 10);
        break;
    case 5:                                      /* RF degradation: throttle */
        n += throttle(s, out[n]);
        break;
    }
    if (n) s->last_action_min = now_min ? now_min : 1;
    return n;
}