import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


def plots(series, save_path=None, title=None):
    title = title.replace('.',',')
    
    n_lines = int(np.ceil(len(series)/2))
    fig, ax = plt.subplots(n_lines,2,figsize=(16,12))
    
    if title is not None:
        fig.suptitle(title, fontsize=16)

    for i, (key, value) in enumerate(series.items()):
        legend = False
        if n_lines==1:
            idx = i%2
        else:
            idx = (i//2,i%2)
        for v in value:
            if isinstance(v, dict):
                x = v['x']
                ax[idx].plot(x,**{k:v[k] for k in v.keys() if k!='x'})
                if 'label' in v:
                    legend = True
            else:
                ax[idx].plot(v)
        if legend:
            ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_title(key)
    
    if save_path is None:
        plt.plot()
    else:
        plt.savefig(save_path)
        plt.close()
        
