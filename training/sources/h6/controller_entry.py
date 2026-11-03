from pathlib import Path
import os,time,ctypes,subprocess
from common import ROOT,write,read,pin_check
from host_reservation import reserve
from pair_queue import main as queue_main,matching_processes

def main():
    pin_check();os.nice(10);assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    subprocess.run(['ionice','-c','2','-n','7','-p',str(os.getpid())],check=True,capture_output=True)
    while True:
        try:reserve(True);break
        except AssertionError as e:
            if str(e) not in ['CPU_ADMISSION','DISK_ADMISSION','RAM_ADMISSION','PSI_ADMISSION']:raise
            write(ROOT/'runtime/QUEUE_STATE.json',{'state':'WAITING_SHARED_HOST_RESOURCES','reason':str(e),'at_unix':time.time()});time.sleep(45)
    try:queue_main()
    finally:
        if not any(matching_processes(ROOT,a) for a in ['core6','vm']):reserve(False)
if __name__=='__main__':main()
