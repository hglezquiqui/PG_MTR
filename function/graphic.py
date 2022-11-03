import matplotlib.pyplot as pp
import seaborn as sbn


def plot_and_reconstructed(dataset, xk, zk):
    pp.figure('Sparse code', figsize=(6, 10))
    pp.clf()
    pp.subplot(311)
    sbn.heatmap(xk, annot=True, cmap='YlGnBu')
    pp.title('Proximal GD Dataset '+dataset['name'][0])
    pp.subplot(312)
    sbn.heatmap(zk, annot=True, cmap='YlGnBu')
    pp.title('Accelerated Proximal GD Dataset '+dataset['name'][0])
    pp.tight_layout()
    pp.draw()
    pp.show()


def plot_objective_function_values(dataset, cri, fobj):
    if len(cri) > 1:
        v = [x for x in range(len(cri))]
        pp.scatter(v, cri, c='b')
        pp.scatter(v, fobj, c='r')
    pp.title(
        'Objective Functions ' + dataset['name'][0] + ' (Blue = PGD, Red = Accelerated PGD)')
    pp.draw()
    pp.pause(0.1)
    pp.show()
