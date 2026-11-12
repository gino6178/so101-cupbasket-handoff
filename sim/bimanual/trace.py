import sys,re,ast
for l in sys.stdin:
  if '[sort]' not in l: continue
  ph=re.search(r'phase=(\[.*?\])',l).group(1); err=ast.literal_eval(re.search(r'err=(\[.*?\])',l).group(1))[0]
  pad=ast.literal_eval(re.search(r'pad=(\[.*?\])',l).group(1)); cube=ast.literal_eval(re.search(r'cube=(\[.*?\])',l).group(1))
  op=re.search(r'open=([0-9.]+)',l).group(1); h=re.search(r'held=\[(\w+)',l).group(1)
  print(ph, f'err={err:.4f}', 'pad',[round(v,4) for v in pad],'cube',[round(v,4) for v in cube],'open',op,h)
