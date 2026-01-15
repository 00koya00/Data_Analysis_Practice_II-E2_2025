
from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as st
import sympy
import parse_data as pdata
import statistics
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

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
    df_numeric = df.select_dtypes(include='number')
    df_numeric = df_numeric.drop('pf', axis=1)
    df_central = df_numeric.apply(lambda x: central(x), axis=0)
    round_dict = 3
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode', 'skew', 'kurtosis'
    df_central.index = row_labels
    display( df_central )
    
def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    df_numeric = df.select_dtypes(include='number')
    df_numeric = df_numeric.drop('pf', axis=1)
    round_dict            = 3
    df_dispersion         = df_numeric.apply(lambda x: dispersion(x), axis=0).round( round_dict )
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

## Setting function for visualization

def color_positive_green(val):
    if val < 0.05:
        color = 'green'
    else:
        color = 'black'
    return 'color: %s' % color

def expand_code(code):
    return "_".join(list(code))

def color_cells(val):
    if abs(val) > 0.5:
        color = 'background-color: salmon'
    elif abs(val) > 0.25:
        color = 'background-color: lightgreen'
    else:
        color = 'background-color: grey'
    return color

## Setting variables

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

xlabels= 'Age', "Mother's education", "Father's education", 'Travel time', 'Study time', 'Failures', 'Freetime', 'Go out', 'Absences'

features= ['age', "Medu", "Fedu", 'travel', 'study', 'failures', 'freetime', 'go_out', 'absences']

ivs    = [age, medu, fedu, ttime, stime, fail, ftime, go, absent]
colors = [cm.hsv(i/9) for i in range (0,9)]

## Setting applied variables

i_0    = y ==0
i_high = y > 0
ab_mean= statistics.mean(absent)
medu_mean= statistics.mean(medu)
fedu_mean= statistics.mean(fedu)
go_mean= statistics.mean(go)

df["pf"]  = [1 if y[i] > 9 else 0 for i in range(0, len(y))]
# pass or fail, calculated by whether its grade is 10 or more. This threshold is following one in prior study.
df["dilig"]  = ["y" if stime[i] > 2 else "n" for i in range(0, len(y))]
# deligence, calculated by whether its study time exceeds the half of the scale: 2.
df["exfail"] = ["y" if fail[i] > 0 else "n" for i in range(0, len(y))]
# experience of failure. Once or more is 1, no failure goes 0
df["part"]   = ["y" if absent[i] > ab_mean else "n" for i in range(0, len(y))]
# participance, calculated by whether its absence exceeds the mean of the population.

df["med_sp"]   = ["y" if medu[i] > medu_mean else "n" for i in range(0, len(y))]
# mother's speciality, calculated by whether its mother's education exceeds the mean of the population.
df["fed_sp"]   = ["y" if fedu[i] > fedu_mean else "n" for i in range(0, len(y))]
# father's speciality, calculated by whether its father's education exceeds the mean of the population.
df["slf_ctrl"] = ["y" if go[i] < go_mean else "n" for i in range(0, len(y))]
# self_control, calculated by whether its time to go out exceeds the mean of the population.


## name for identical group
df["group"] = df["dilig"] + "_" + df["exfail"] + "_" + df["part"]
df["group_2"] = df["med_sp"] + "_" + df["fed_sp"] + "_" + df["slf_ctrl"]

## make up name list

binary_vars = [
    'dilig', 'exfail', 'part',
    'med_sp', 'fed_sp', 'slf_ctrl'
]

numeric_vars = [
    'age', 'Medu', 'Fedu', 'travel', 'study',
    'failures', 'freetime', 'go_out', 'absences'
]

classif_vars = [
    'pf', 'age', 'Medu', 'Fedu', 'travel', 'study',
    'failures', 'freetime', 'go_out', 'absences'
]

X_numeric = df[numeric_vars]

group_vars = ['group', 'group_2']

## making test data

# for continuum grade

X = df[numeric_vars]
y = df["Grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# for category

Xc = df[classif_vars]
yc = df["pf"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.3, random_state=1
)

# for high-grade prediction

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# for low-grade prediction

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X, y, test_size=0.3, random_state=0
)


## for improvements

# scaling

xtest_scaled  =pd.DataFrame()
xtrain_scaled =pd.DataFrame()

scaler_pf = preprocessing.StandardScaler()
scaler_pf.fit(Xc_train[numeric_vars])
xtrain_scaled = pd.DataFrame(scaler_pf.transform(Xc_train[numeric_vars]), columns=numeric_vars)
xtest_scaled  = pd.DataFrame(scaler_pf.transform(Xc_test[numeric_vars]),  columns=numeric_vars)

y_train_pca = yc_train.reset_index(drop=True)
y_test_pca  = yc_test.reset_index(drop=True)

scaler_hl = preprocessing.StandardScaler()
scaler_hl.fit(X_train_h[numeric_vars])
X_train_hl_scaled = scaler_hl.transform(X_train_h[numeric_vars])
X_test_hl_scaled  = scaler_hl.transform(X_test_h[numeric_vars])


y_train_high_pca = (y_train_h >= 15).astype(int).reset_index(drop=True)
y_test_high_pca  = (y_test_h  >= 15).astype(int).reset_index(drop=True)
y_train_low_pca = (y_train_l <= 5).astype(int).reset_index(drop=True)
y_test_low_pca  = (y_test_l  <= 5).astype(int).reset_index(drop=True)

def plot_descriptive():

    display_title('Correlations amongst main variables.', pref='Figure', num=1)
    fig,axs = plt.subplots( 3, 3, figsize=(10,10), tight_layout=True)
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
