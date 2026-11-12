import sys, math, collections
# best-of-tilts map: for each cell and arm, the best share reached within 1 cm over the tilt runs given
runs=sys.argv[1:]
cell=collections.defaultdict(lambda: collections.defaultdict(list))
for ri,f in enumerate(runs):
    for l in open(f,errors='ignore'):
        if not l.startswith('BITIDY_REACH'): continue
        x,y,z,el,er=map(float,l.split()[1:6])
        k=(int((x-0.02)//0.04),int((y+0.22)//0.04),0 if z<0.09 else (1 if z<0.13 else 2))
        if not math.isnan(el): cell[k][('L',ri)].append(el<0.01)
        if not math.isnan(er): cell[k][('R',ri)].append(er<0.01)
def best(k,a):
    v=[sum(cell[k][(a,ri)])/len(cell[k][(a,ri)]) for ri in range(len(runs)) if cell[k][(a,ri)]]
    return max(v) if v else None
for zb,name in enumerate(("0.05-0.09","0.09-0.13","0.13-0.16")):
    print(f"z {name}  y: " + " ".join(f"{-0.20+0.04*i:+.2f}" for i in range(11)))
    for xi in range(8):
        line=f"x {0.04+0.04*xi:.2f}      "
        for yi in range(11):
            L,R=best((xi,yi,zb),'L'),best((xi,yi,zb),'R')
            f=lambda v:'-' if v is None else str(min(9,int(v*10)))
            line+=f" {f(L)}{f(R)}  "
        print(line)
