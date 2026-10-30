"""Create synthetic seismic-like sections for software testing only."""
from pathlib import Path
import numpy as np

def main():
    rng=np.random.default_rng(17); xs=[]; ys=[]; groups=[]
    h,w=128,160
    for g in range(4):
        for _ in range(10):
            z=np.linspace(0,10*np.pi,h)[:,None]
            x=np.linspace(0,2*np.pi,w)[None,:]
            phase=.45*np.sin((1.2+.15*g)*x)+.10*np.sin(4*x)
            arr=np.sin(z+phase)+.25*np.sin(2*z+1.2*phase)+rng.normal(0,.18,(h,w))
            mask=np.zeros((h,w),np.uint8)
            fx=int(rng.integers(w//4,3*w//4))
            throw=int(rng.integers(4,12))
            # Artificial reflector discontinuity; this is not a geological simulation.
            arr[:,fx:]=np.roll(arr[:,fx:],throw,axis=0)
            width=int(rng.integers(1,3))
            mask[:,max(0,fx-width):min(w,fx+width+1)]=1
            xs.append(arr.astype("float32")); ys.append(mask); groups.append(f"SYNTHETIC_BLOCK_{g}")
    out=Path("data/sample/synthetic_demo.npz"); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,seismic=np.stack(xs),fault_mask=np.stack(ys),groups=np.asarray(groups))
    print(f"Saved {out} — synthetic demonstration only.")

if __name__=="__main__": main()
