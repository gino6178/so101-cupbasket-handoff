import sys, math
rows=[l.split()[1:] for l in open(sys.argv[1], errors='ignore') if l.startswith('BITIDY_ROW')]
bx, by = -0.05, float(sys.argv[2])
bins={}
for r in rows:
    sx,sy,gx,gy=map(float,r[:4]); g,l,o,p,s=map(int,r[4:9])
    d=math.hypot(sx-bx, sy-by); gd=math.hypot(gx-bx,gy-by)
    k=(round(d*50)/50, round(gd,2))
    b=bins.setdefault(k,[0,0,0,0,0]); b[0]+=1; b[1]+=g; b[2]+=l; b[3]+=o; b[4]+=s
print("cube_dist goal_dist  n  grasp lift over success")
for k in sorted(bins):
    n,g,l,o,s=bins[k]; print(f"{k[0]:.2f}  {k[1]:.2f}  {n:4d}  {g/n:.2f} {l/n:.2f} {o/n:.2f} {s/n:.2f}")
