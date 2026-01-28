import descriptive as desc
from IPython.display import display,Markdown #,HTML
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as st
import sympy
import scipy.stats as st
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

X_2d = desc.Xc_train.iloc[:, [2, 9]]
X_train_2d = desc.Xc_train.loc[:, ['Medu', 'absences']]
X_test_2d  = desc.Xc_test.loc[:, ['Medu', 'absences']]

def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
    nlabels   = np.unique( labels ).size
    colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
    ax        = plt.gca() if (ax is None) else ax
    xmin,xmax = x.min(axis=0), x.max(axis=0)
    Xp,Yp     = np.meshgrid( np.linspace(xmin.iloc[0],xmax.iloc[0],n) , np.linspace(xmin.iloc[1],xmax.iloc[1],n) )
    xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
    xp_df = pd.DataFrame(xp, columns=x.columns) 
    labelsp   = classifier.predict(xp_df)
    Labelsp   = np.reshape(labelsp, Xp.shape)
    cmap      = ListedColormap(colors)
    for i,label in enumerate( np.unique(labels) ):
        xx   = x[labels==label]
        ax.scatter( xx.iloc[:,0], xx.iloc[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}' )
    plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

def decition_surface(num=1):
    mlp2 = make_pipeline(
    StandardScaler(),
    MLPClassifier(solver='lbfgs', alpha=0.0001,
                  hidden_layer_sizes=(50, 20),
                  random_state=0, max_iter=5000)
    )
    mlp2.fit(X_train_2d, desc.yc_train)
    pred2 = mlp2.predict(X_test_2d)
    plt.figure(figsize=(8,8))
    plot_decision_surface(mlp2, X_2d, desc.yc_train, colors = ['y', 'g'])
    plt.plot(desc.Xc_test.loc[:,'Medu'], desc.Xc_test.loc[:,'absences'], 'ko', label='Test set')
    desc.display_title("Pass/ Fail Distribution in 2D: Mother’s Education vs Absences (with test set)", pref='Figure', num=num)
    plt.legend(["Fail", "Pass", "Test set"])
    plt.ylabel("Absence (days)")
    plt.xlabel("Mother's education (0=lower, 4=higher)")
    plt.show()

def decision_feature(num=1):
    y_train_high_binary = (desc.y_train_h >= 15).astype(int)
    y_test_high_binary  = (desc.y_test_h  >= 15).astype(int)
    y_train_low_binary = (desc.y_train_l <= 5).astype(int)
    y_test_low_binary  = (desc.y_test_l <= 5).astype(int)
    dt_high = DecisionTreeClassifier(max_depth=3, random_state=0)
    dt_high.fit(desc.X_train_h, y_train_high_binary) 
    dt_low  = DecisionTreeClassifier(max_depth=3, random_state=0)
    dt_low.fit(desc.X_train_l, y_train_low_binary) 
    imp_high = pd.Series(dt_high.feature_importances_, index=desc.features)
    imp_low  = pd.Series(dt_low.feature_importances_, index=desc.features)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    desc.display_title("Predictors for High/ Low-Achievers", pref='Figure', num=num)
    imp_high.sort_values().plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Predictors for High-Achievers (G3 >= 15)')
    ax1.set_xlabel('Feature Importance')
    imp_low.sort_values().plot(kind='barh', ax=ax2, color='salmon')
    ax2.set_title('Predictors for Low-Achievers (G3 <= 5)')
    ax2.set_xlabel('Feature Importance')
    plt.tight_layout()
    return plt.show()