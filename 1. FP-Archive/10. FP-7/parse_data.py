import pandas as pd
from sklearn import preprocessing

df0 = pd.read_csv('student-mat.csv', sep = ';')

## General use data
df = df0[  ['school', 'age', 'Medu', 'Fedu', 'traveltime','studytime','failures', 'freetime', 'goout', 'absences', 'G3']  ]
df = df.rename( columns={'traveltime':'travel', 'studytime':'study', 'goout':'go_out', 'G3':'Grade'} )


