import json, struct, sys, numpy as np
exec(open('emu_count.py').read().split("call(\"bench_init\")")[0])
call("bench_init")
frame_addr = syms["g_frame"]
symtab=[(s["st_value"]&~1, s["st_size"], s.name) for s in elf.get_section_by_name(".symtab").iter_symbols() if s["st_info"]["type"]=="STT_FUNC" and s["st_size"]>0]
rng = np.random.default_rng(7); N=1500
p = ts.make_scenarios(1, "NOM", N, rng); y, narc, Tp, _ = ts.simulate(p, N, rng)
for t in range(N):
    uc.mem_write(frame_addr, struct.pack("<6fB3x", *(float(v) for v in y[0, :, t]), float(Tp[0, t]), 0))
    if t < N-1: call("bench_step")
by={}
def hk(u,addr,size,ud):
    for a,s,n in symtab:
        if a<=addr<a+s: by[n]=by.get(n,0)+1; return
    by['?']=by.get('?',0)+1
uc.reg_write(UC_ARM_REG_SP, SP0); uc.reg_write(UC_ARM_REG_LR, STOP|1)
h=uc.hook_add(UC_HOOK_CODE,hk); uc.ctl_flush_tb(); uc.emu_start(syms["bench_step"]|1, STOP)
tot=sum(by.values()); print(tot, sorted(by.items(), key=lambda x:-x[1]))
json.dump(dict(total=tot, by=by), open('emu_breakdown.json','w'))
