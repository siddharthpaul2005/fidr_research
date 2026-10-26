/* prints the command frames generated for every action, plus safety-rule checks */
#include <stdio.h>
#include "fdir_cmd.h"
static const char *nm[] = {"", "GRID -> Va correction", "CATH -> switch neutraliser",
                           "ARC -> throttle + HV conditioning", "FEED -> heater set point",
                           "RF -> throttle"};
int main(void)
{
    uint8_t out[4][8];
    for (int a = 1; a <= 5; a++) {
        cmd_state_t s; cmd_init(&s);
        int n = cmd_encode(&s, a, 6000, out);
        printf("%-34s %d frame(s):", nm[a], n);
        for (int i = 0; i < n; i++) { printf("  ["); for (int b = 0; b < 8; b++) printf("%02X%s", out[i][b], b < 7 ? " " : ""); printf("]"); }
        printf("\n");
    }
    cmd_state_t s; cmd_init(&s);
    cmd_encode(&s, 1, 6000, out);
    int n1 = cmd_encode(&s, 1, 6060, out);
    printf("safety: 2nd action 1 h later -> %d frames (blocked by 6 h gap)\n", n1);
    int seq[] = {7000, 7400, 7800, 8200, 8600};
    printf("safety: repeated grid corrections every 6.7 h ->");
    for (int i = 0; i < 5; i++) printf(" %d", cmd_encode(&s, 1, seq[i], out));
    printf("  (Va now %d V, safe_mode=%d)\n", s.va_V, s.safe_mode);
    cmd_state_t s2; cmd_init(&s2); cmd_encode(&s2, 2, 6000, out);
    int n2 = cmd_encode(&s2, 2, 7000, out);
    printf("safety: 2nd neutraliser switch -> %d frames, safe_mode=%d (no spare left)\n", n2, s2.safe_mode);
    uint8_t chk[] = {'1','2','3','4','5','6','7','8','9'};
    printf("CRC-16/CCITT-FALSE self-test (\"123456789\" -> 0x29B1): 0x%04X\n", crc16_ccitt(chk, 9));
    return 0;
}