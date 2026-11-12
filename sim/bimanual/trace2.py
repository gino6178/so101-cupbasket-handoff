import sys,re,ast
mn=int(sys.argv[1])
for l in sys.stdin:
  if '[sort]' not in l: continue
  ph=ast.literal_eval(re.search(r'phase=(\[.*?\])',l).group(1))
  if ph[0]<mn: continue
  err=ast.literal_eval(re.search(r'err=(\[.*?\])',l).group(1))[0]
  pad=ast.literal_eval(re.search(r'pad=(\[.*?\])',l).group(1)); cube=ast.literal_eval(re.search(r'cube=(\[.*?\])',l).group(1)); goal=ast.literal_eval(re.search(r'goal=(\[.*?\])',l).group(1))
  op=re.search(r'open=([0-9.]+)',l).group(1); h=re.search(r'held=\[(\w+)',l).group(1)
  print(ph[0], f'err={err:.4f}', 'pad',[round(v,3) for v in pad],'cube',[round(v,3) for v in cube],'goal',[round(v,2) for v in goal],'open',op,h)
