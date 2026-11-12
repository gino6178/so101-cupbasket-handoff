import numpy as np
d=np.load("/home/"+__import__("getpass").getuser()+"/demos_LidPlace.npz")
a=d["act"];o=d["obs"]
print("act shape",a.shape,"obs",o.shape)
print("sat frac per dim",np.round((np.abs(a)>=0.999).mean(0),3))
print("act mean",np.round(a.mean(0),2));print("act std",np.round(a.std(0),2))
