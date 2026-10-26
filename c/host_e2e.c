/* End-to-end host test: raw telemetry frames -> fdir_step. frames.bin: N x (6 float + 1 float narc) */
#include <stdio.h>
#include "fdir.h"
int main(void)
{
    FILE *fi = fopen("frames.bin", "rb"), *fo = fopen("e2e_out.bin", "wb");
    if (!fi || !fo) return 1;
    static fdir_t f; fdir_init(&f);
    float v[7]; long k = 0; int act_t = -1, act = 0;
    while (fread(v, sizeof(float), 7, fi) == 7) {
        fdir_frame_t fr = { v[0], v[1], v[2], v[3], v[4], v[5], (uint8_t)v[6] };
        fdir_action_t a = fdir_step(&f, &fr, -200.0f);
        if (a && act_t < 0) { act_t = (int)k; act = a; }
        float o[3] = { fdir_dbg_cusum(&f), fdir_dbg_kf(&f), fdir_dbg_bocpd(&f) };
        fwrite(o, sizeof(float), 3, fo);
        k++;
    }
    printf("%d %d %d %d %d %d %d\n", (int)f.t_cusum, (int)f.t_kf, (int)f.t_bocpd, (int)f.t_arc,
           (int)f.t_alarm, act_t, act);
    return 0;
}
