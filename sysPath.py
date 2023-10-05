import os
import sys
cfd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cfd)))
sys.path.append(cfd+'/WB')

if __name__ == '__main__': 
    print("-------------- if __name__ == '__main__': -----in sysPath.py-----------")
    print("cfd = ",cfd)
    print("sys.path = ",sys.path)
