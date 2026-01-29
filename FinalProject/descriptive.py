import statistics
import numpy as np
import pandas as pd
from scipy import stats
import parse_data as pdata
from sklearn import preprocessing
from matplotlib import pyplot as plt
from IPython.display import display,Markdown #,HTML
from sklearn.model_selection import train_test_split

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

def display_summary_table(num=1):
    display_title('Summary statistics of numerical variables.', pref='Table', num=num, center=False)
    df_numeric = df.select_dtypes(include='number')
    summary_df = df_numeric.describe()
    summary_df = summary_df.rename(index={
        'mean': 'Mean',
        'std': 'Std. Dev.',
        'min': 'Min',
        '50%': 'Median',
        'max': 'Max'
    })
    summary_df = summary_df.rename(columns={
        'Medu': "Mother's Education",
        'Fedu': "Father's education",
        'go_out': 'Going out',
    })
    summary_df = summary_df.drop('pf', axis=1)
    summary_df = summary_df.drop('25%')
    summary_df = summary_df.drop('75%')
    summary_df = summary_df.drop('count')
    display(summary_df.round(3))

def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)

def expand_code(code):
    return "_".join(list(code))

def translate_one(s, pairs_long):
    result = []
    for i, ch in enumerate(s):
        if ch == 'n':
            chosen = pairs_long[i][1]
        else: 
            chosen = pairs_long[i][0]
        result.append(chosen)
    return "_".join(result)

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

xlabels= 'Age(years)', "Mother's education (0=lower, 4=higher)", "Father's education (0=lower, 4=higher)", 'Travel time (1=none, 4=more)', 'Study time (1=shorter, 4=longer)', 'Failures (0=none, 3=many)', 'Freetime (0=short, 5=many)', 'Go out (1=less, 4=more)', 'Absences (days)'

features= ['age', "Medu", "Fedu", 'travel', 'study', 'failures', 'freetime', 'go_out', 'absences']

ivs    = [age, medu, fedu, ttime, stime, fail, ftime, go, absent]
colors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#7F7F7F','#BCBD22']

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
df["part"]   = ["y" if absent[i] < ab_mean else "n" for i in range(0, len(y))]
# participance, calculated by whether its absence exceeds the mean of the population.

df["med_sp"]   = ["y" if medu[i] > medu_mean else "n" for i in range(0, len(y))]
# mother's speciality, calculated by whether its mother's education exceeds the mean of the population.
df["fed_sp"]   = ["y" if fedu[i] > fedu_mean else "n" for i in range(0, len(y))]
# father's speciality, calculated by whether its father's education exceeds the mean of the population.
df["slf_ctrl"] = ["y" if go[i] < go_mean else "n" for i in range(0, len(y))]
# self_control, calculated by whether its time to go out exceeds the mean of the population.



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

## name for identical group
df["group"] = df["dilig"] + "_" + df["exfail"] + "_" + df["part"]
df["group_2"] = df["med_sp"] + "_" + df["fed_sp"] + "_" + df["slf_ctrl"]

## Setting variables for classical

nnn   = df[df["group"] == "n_n_n"]
nny   = df[df["group"] == "n_n_y"]
nyn   = df[df["group"] == "n_y_n"]
nyy   = df[df["group"] == "n_y_y"]
ynn   = df[df["group"] == "y_n_n"]
yny   = df[df["group"] == "y_n_y"]
yyn   = df[df["group"] == "y_y_n"]
yyy   = df[df["group"] == "y_y_y"]


site1 = np.array(nnn["Grade"])
site2 = np.array(nny["Grade"])
site3 = np.array(nyn["Grade"])
site4 = np.array(nyy["Grade"])
site5 = np.array(ynn["Grade"])
site6 = np.array(yny["Grade"])
site7 = np.array(yyn["Grade"])
site8 = np.array(yyy["Grade"])

about = df.groupby("group")["Grade"].agg(["mean", "std", "count"])
about["se"] = about["std"] / np.sqrt(about["count"])
order = ["n_n_n", "n_n_y", "n_y_n", "n_y_y", "y_n_n", "y_n_y", "y_y_n", "y_y_y"]



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

def plot_descriptive(keep=('Medu', 'Fedu', 'failures', 'absences'), num=1):
    keep_set = set(keep)
    selected = [(x, c, xlabel, feat) for x, c, xlabel, feat in zip(ivs, colors, xlabels, features)
            if feat in keep_set]
    display_title('Correlations amongst main variables.', pref='Figure', num=num)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    axs = axs.ravel()

    for ax, (x, c, xlabel, feat) in zip(axs, selected):
        ax.scatter(x[i_high], y[i_high], alpha=0.2, color=c)
        plot_regression_line(ax, x[i_high], y[i_high], color='k', ls='-', lw=2)

        r = corrcoeff(x[i_high], y[i_high])        
        ax.text(0.65, 0.12, f'r = {r:.3f}\nr when grade > 0', color=c,
        transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
        
        ax.scatter(x[i_0], y[i_0], alpha=0.5, color='0.4')
        ax.set_xlabel(xlabel) 

        if feat.lower() == 'absences':
            ax.set_xticks([0, 20, 40, 60])

    axs[0].set_ylabel('Grade (0-20 scale)')
    axs[2].set_ylabel('Grade (0-20 scale)')

    for ax, s in zip(axs, ['a', 'b', 'c', 'd']):
        ax.text(0.92, 0.92, f'({s})', size=12, transform=ax.transAxes)

    plt.show()

