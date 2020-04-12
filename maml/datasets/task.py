from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt

Task = namedtuple('Task', ['x', 'y', 'task_info'])

def plot_task(task):
    fig, ax = plt.subplots(5, task.x.size(0) // 5, figsize=(5,5*1.3))
    images = task.x.cpu().numpy().transpose(0,2,3,1)
    for i in range(5):
        for j in range(task.x.size(0) // 5):
            ax[i][j].imshow(images[5 * i + j])
            ax[i][j].set_title("Class: {}".format(task.y[5 * i + j].item()))
            ax[i][j].set_axis_off()
