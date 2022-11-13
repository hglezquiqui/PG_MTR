import TestingModel as test
import sys
from data.RepositoryEnum import RepositoryEnum as RepEnum

if __name__ == '__main__':
    if(sys.argv[1] == 'MTR_EA'):
        test.MTR_EA()
    elif(sys.argv[1] == 'GMLR1'):
        dataset = {'name': [sys.argv[2]], 'output': [RepEnum[sys.argv[2]].value]}
        test.GMLR1(dataset)