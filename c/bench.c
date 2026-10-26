/* Bare-metal harness for instruction counting under emulation (Cortex-M4F). */
#include "fdir.h"
fdir_t g_fdir;
volatile fdir_frame_t g_frame;
volatile float g_va = -200.0f;
volatile int g_action;
void fdir_detect_z(fdir_t *f, const float z[FDIR_NCH], uint8_t n_recycle);

void bench_init(void) { fdir_init(&g_fdir); }
void bench_step(void) { g_action = fdir_step(&g_fdir, (const fdir_frame_t *)&g_frame, g_va); }

extern unsigned _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);
void Reset_Handler(void)
{
    /* enable FPU: CPACR CP10/CP11 full access */
    *(volatile unsigned *)0xE000ED88 |= (0xFu << 20);
    unsigned *s = &_sidata, *d = &_sdata;
    while (d < &_edata) *d++ = *s++;
    for (d = &_sbss; d < &_ebss;) *d++ = 0;
    main();
    for (;;) ;
}
volatile int g_run;
int main(void) { bench_init(); while (g_run) bench_step(); return 0; }
__attribute__((section(".isr_vector"), used))
const void *vectors[2] = { &_estack, (void *)Reset_Handler };
