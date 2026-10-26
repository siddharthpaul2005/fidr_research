/* Command interface: onboard computer (OBC) -> thruster power unit (PPU).
 * Every autonomous action is sent as one or more fixed 8-byte frames:
 *
 *   byte 0-1 : sync word 0xEB 0x90
 *   byte 2   : sequence counter (increments every frame, lets the PPU drop duplicates)
 *   byte 3   : opcode
 *   byte 4-5 : signed 16-bit argument, little-endian (units depend on opcode)
 *   byte 6-7 : CRC-16/CCITT-FALSE over bytes 0-5 (big-endian)
 *
 * The PPU checks sync + CRC + sequence + argument range, executes, and reports
 * the result in its next telemetry frame (ack/nack flag + the new set point).  */
#ifndef FDIR_CMD_H
#define FDIR_CMD_H
#include <stdint.h>

enum {
    OP_SET_VA        = 0x10,  /* accelerator-grid voltage, arg = volts (e.g. -225)          */
    OP_SEL_NEUTRAL   = 0x11,  /* select neutraliser/cathode unit, arg = 0 (A) or 1 (B)       */
    OP_SET_BEAM      = 0x12,  /* beam-current set point, arg = 0.1 mA units (150 = 15.0 mA)  */
    OP_HV_CONDITION  = 0x13,  /* high-voltage conditioning, arg = number of recycle cycles   */
    OP_FEED_HEATER   = 0x14   /* sublimator heater set-point change, arg = 0.1 degC units    */
};

/* safety limits enforced by the OBC before sending (the PPU re-checks them) */
#define CMD_VA_MIN_V        (-300)   /* most negative accel voltage the PPU supports  */
#define CMD_VA_MAX_V        (-150)
#define CMD_BEAM_MIN_dmA    (90)     /* 9 mA: lower end of the BIT-3 beam-current range */
#define CMD_BEAM_MAX_dmA    (170)    /* 17 mA: upper end                                */
#define CMD_HEATER_MAX_ddC  (20)     /* at most +2.0 degC per autonomous step           */
#define CMD_MIN_GAP_MIN     (360)    /* at most one autonomous action per 6 h            */
#define CMD_MAX_PER_CLASS   (3)      /* after 3 actions for the same fault -> safe mode, wait for ground */

typedef struct {
    int16_t va_V;          /* current accelerator voltage set point          */
    int16_t beam_dmA;      /* current beam-current set point (0.1 mA)        */
    uint8_t neutraliser;   /* 0 = A, 1 = B                                    */
    uint8_t seq;           /* frame sequence counter                          */
    uint32_t last_action_min;
    uint8_t n_actions[6];  /* actions taken per fault class                   */
    uint8_t safe_mode;     /* 1 = autonomy suspended, waiting for ground      */
} cmd_state_t;

void cmd_init(cmd_state_t *s);
uint16_t crc16_ccitt(const uint8_t *d, int n);
/* Translate an FDIR action into command frames.  Returns number of 8-byte
 * frames written to out (0 if blocked by a safety rule). */
int cmd_encode(cmd_state_t *s, int action, uint32_t now_min, uint8_t out[][8]);
#endif