from utils import *
import matplotlib.pyplot as plt

paths = {
'resnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resnet20/log.txt', 
'resattnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resattnet20/log.txt', 
}

field = ['Valid Acc.']

monitor = LoggerMonitor(paths)
monitor.plot(names=field)

savefig('test.eps')
plt.show()