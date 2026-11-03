"""CPU diagnostics for predeclared16 EMA epochs plus fixed EMA SWA, both arms."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,fcntl,subprocess,sys,time
from common import ROOT,write,pin_check
from diagnostic_receipts import expected_binding,receipts_complete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gold-labels',required=True);ap.add_argument('--once',action='store_true');a=ap.parse_args();package=pin_check()
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True);lock=(runtime/'observe.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=[f'epoch{e:02d}_ema.pt' for e in range(1,17)]+['swa_ema05_10.pt']
    while True:
        checked=0
        for arm in ['core6','vm']:
            for name in names:
                checkpoint=ROOT/'outputs'/arm/name;folder=runtime/'diagnostics'/arm/checkpoint.stem
                if not checkpoint.exists():continue
                binding=expected_binding(ROOT,arm,checkpoint,package)
                if receipts_complete(ROOT,folder,binding):checked+=1;continue
                p=subprocess.run([sys.executable,str(ROOT/'diagnostics.py'),'--arm',arm,'--checkpoint',str(checkpoint),'--gold-labels',a.gold_labels,'--fm-holdout'])
                if p.returncode==75:break
                if p.returncode:raise RuntimeError('DIAGNOSTIC_HARD_ERROR:'+arm+':'+name+':'+str(p.returncode))
                checked+=1
        write(runtime/'OBSERVER_STATE.json',{'state':'COMPLETE' if checked==34 else 'OBSERVING','records':checked,'expected':34,'at_unix':time.time()})
        if a.once or checked==34:return
        time.sleep(45)


if __name__=='__main__':main()
