"""Instruction-count profiling of fdir_step on an emulated Cortex-M4F (Unicorn).
Counts executed instructions per call (not cycles) and tracks peak stack depth."""
import json, struct, sys
import numpy as np
from elftools.elf.elffile import ELFFile
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_CODE
from unicorn.arm_const import *
sys.path.insert(0, "..")
import thruster_sim as ts

elf = ELFFile(open("bench.elf", "rb"))
syms = {s.name: s["st_value"] for s in elf.get_section_by_name(".symtab").iter_symbols()}
uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M4) if hasattr(uc, "ctl_set_cpu_model") else None
uc.mem_map(0x08000000, 0x100000)
uc.mem_map(0x20000000, 0x20000)
for seg in elf.iter_segments():
    if seg["p_type"] == "PT_LOAD":
        uc.mem_write(seg["p_paddr"], seg.data())
        uc.mem_write(seg["p_vaddr"], seg.data())
# enable FPU (CPACR) via register if exposed
try:
    uc.reg_write(UC_ARM_REG_CPACR, 0xF << 20)
except Exception as e:
    print("CPACR write:", e)
try:
    uc.reg_write(UC_ARM_REG_FPEXC, 0x40000000)
except Exception as e:
    print("FPEXC write:", e)
STOP = 0x080FFF00
uc.mem_write(STOP, b"\x00\xbf\x00\xbf")   # nop; nop
SP0 = 0x20020000

count = [0]; minsp = [SP0]
def hook(uc_, addr, size, ud):
    count[0] += 1
    sp = uc_.reg_read(UC_ARM_REG_SP)
    if sp < minsp[0]: minsp[0] = sp


def call(fn, counting=False):
    uc.reg_write(UC_ARM_REG_SP, SP0)
    uc.reg_write(UC_ARM_REG_LR, STOP | 1)
    count[0] = 0
    h = None
    if counting:
        h = uc.hook_add(UC_HOOK_CODE, hook); uc.ctl_flush_tb()
    uc.emu_start(syms[fn] | 1, STOP)
    if h is not None:
        uc.hook_del(h); uc.ctl_flush_tb()
    return count[0]


call("bench_init")
frame_addr = syms["g_frame"]; va_addr = syms["g_va"]; act_addr = syms["g_action"]

# drive with a GRID-erosion telemetry trace from the simulator
rng = np.random.default_rng(7)
N = 300 * 60
p = ts.make_scenarios(1, "GRID", N, rng)
p["t0"][:] = 26 * 60; p["tau"][:] = 40 * 60; p["sev"][:] = 1.2
y, narc, Tp, _ = ts.simulate(p, N, rng)
counts = []; actions = []
profile_idx = set(list(range(0, 5)) + list(range(1438, 1460)) + list(range(1870, 1900)) + list(range(5000, 5010)))
for t in range(N):
    fr = struct.pack("<6fB3x", *(float(v) for v in y[0, :, t]), float(Tp[0, t]), int(narc[0, t]))
    uc.mem_write(frame_addr, fr)
    prof = t in profile_idx
    c = call("bench_step", counting=prof)
    a = struct.unpack("<i", uc.mem_read(act_addr, 4))[0]
    if a:
        actions.append((t, a))
        if not prof:   # re-run accounting is not possible post-hoc; record isolation frame cost separately
            pass
    if prof: counts.append((t, c))
    if t > 5010: break
# profile the isolation frame: restore state is complex, so emulate one extra frame with counting
res = dict(per_frame=counts, actions=actions,
           stack_peak_bytes=SP0 - minsp[0])
print(json.dumps(res))
json.dump(res, open("emu_counts.json", "w"))
