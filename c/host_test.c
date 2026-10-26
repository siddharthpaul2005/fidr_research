/* Host equivalence test: feeds a normalised-residual trace (z.bin: N x 5 float32,
 * narc.bin: N uint8) through fdir_detect_z and writes per-sample debug taps. */
#include <stdio.h>
#include <stdlib.h>
#include "fdir.h"
void fdir_detect_z(fdir_t *f, const float z[FDIR_NCH], uint8_t n_recycle);

int main(int argc, char **argv)
{
    FILE *fz = fopen("z.bin", "rb"), *fn = fopen("narc.bin", "rb"), *fo = fopen("c_out.bin", "wb");
    if (!fz || !fn || !fo) return 1;
    static fdir_t f;
    fdir_init(&f);
    float z[5]; uint8_t n; long k = 0;
    while (fread(z, sizeof(float), 5, fz) == 5 && fread(&n, 1, 1, fn) == 1) {
        fdir_detect_z(&f, z, n);
        float o[4] = { fdir_dbg_cusum(&f), fdir_dbg_kf(&f), fdir_dbg_bocpd(&f), f.arc };
        fwrite(o, sizeof(float), 4, fo);
        k++;
    }
    printf("samples %ld  alarms: cusum %d kf %d bocpd %d arc %d fused %d\n", k,
           (int)f.t_cusum, (int)f.t_kf, (int)f.t_bocpd, (int)f.t_arc, (int)f.t_alarm);
    printf("sizeof(fdir_t) = %u bytes\n", (unsigned)sizeof(fdir_t));
    fclose(fo);
    return 0;
}
