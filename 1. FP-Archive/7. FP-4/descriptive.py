
## importing modules
from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as st
import sympy
from sklearn.datasets import load_iris
import scipy.stats as st
import parse_data as pdata


## defining functions
def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )

def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    x3     = st.skew(x)
    x4     = st.kurtosis(x)
    return x0, x1, x2, x3, x4

def dispersion(x, print_output=True):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    return y0,y1,y2,y3,y4,y5,y6

def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df.apply(lambda x: central(x), axis=0)
    round_dict = 3
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode', 'skew', 'kurtosis'
    df_central.index = row_labels
    display( df_central )
    
def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = 3
    df_dispersion         = df.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)

# Setting variables before analysis
df = pdata.df

y    = df['Grade']

age  = df['age']
medu = df['Medu']
fedu = df['Fedu']
ttime  =df['travel']
stime  =df['study']
fail   =df['failures']
ftime  =df['freetime']
go     =df['go_out']
absent =df['absences']

columns=df.columns

xlabels = 'Age', "Mother's education", "Father's education", 'Travel time', 'Study time', 'Failures', 'Freetime', 'Go out', 'Absences'  

i_0       = y ==0
i_high    = y > 0
ivs     = [age, medu, fedu, ttime, stime, fail, ftime, go, absent]
colors = [cm.hsv(i/9) for i in range (0,9)]


# Plot function

def plot_descriptive():
    
    fig,axs = plt.subplots( 3, 3, figsize=(10,10), tight_layout=True )
    axx       = [axs[1, 2], axs[2, 2]]
    
    for ax,x,c in zip(axs.ravel(), ivs, colors):
        ax.scatter( x[i_high], y[i_high], alpha=0.2, color=c )
        plot_regression_line(ax, x[i_high], y[i_high], color='k', ls='-', lw=2)
        r   = corrcoeff(x[i_high], y[i_high])
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    for ax,x,c in zip(axs.ravel(), ivs, colors):
        ax.scatter( x[i_0], y[i_0], alpha=0.5)

    [ax.set_xlabel(s) for ax,s in zip(axs.flat,xlabels)]
    axs[2, 2].set_xticks([0, 20, 40, 60])
    [ax.set_ylabel('Grade') for ax in axs[:,0]]
    for a, s in zip(axs.flat, xlabels):
        a.set_yticklabels([])
    
    [axs[1, 2].plot(fail[y==q].mean(), q, 'o', color=c, mfc='w', ms=5)  for q in range (0, 19)]
    [axs[2, 2].plot(absent[y==q].mean(), q, 'o', color=c, mfc='w', ms=5)  for q in range (0, 19)]

    panel_labels = 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'
    [ax.text(0.92, 0.92, f'({s})', size=12, transform=ax.transAxes)  for ax,s in zip(axs.ravel(), panel_labels)]
    plt.show()
    
    display_title('Correlations amongst main variables.', pref='Figure', num=1)
