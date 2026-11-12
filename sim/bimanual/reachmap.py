import sys, math
rows=[list(map(float,l.split()[1:6])) for l in open(sys.argv[1],errors='ignore') if l.startswith('BITIDY_REACH')]
print(len(rows),"points; cell = left/right share reached within 1 cm (0-9), 4x4 cm cells, by height band")
for zlo,zhi in ((0.05,0.09),(0.09,0.13),(0.13,0.16)):
    print(f"z {zlo:.2f}-{zhi:.2f}  y: " + " ".join(f"{y:+.2f}" for y in [(-0.22+0.04*i+0.02) for i in range(11)]))
    for xi in range(8):
        x0=0.02+0.04*xi; line=f"x {x0+0.02:.2f}      "
        for yi in range(11):
            y0=-0.22+0.04*yi
            c=[r for r in rows if x0<=r[0]<x0+0.04 and y0<=r[1]<y0+0.04 and zlo<=r[2]<zhi]
            L=[r[3] for r in c if not math.isnan(r[3])]; R=[r[4] for r in c if not math.isnan(r[4])]
            f=lambda v: str(min(9,int(10*sum(e<0.01 for e in v)/len(v)))) if v else '-'
            line+=f" {f(L)}{f(R)}  "
        print(line)
