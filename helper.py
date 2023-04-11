import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(waiting_time):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(waiting_time)
    #plt.plot(mean_waiting_time)
    plt.ylim(ymin=0)
    plt.text(len(waiting_time)-1, waiting_time[-1], str(waiting_time[-1]))
    #plt.text(len(mean_waiting_time)-1, mean_waiting_time[-1], str(mean_waiting_time[-1]))
    plt.show(block=False)
    plt.pause(.1)
