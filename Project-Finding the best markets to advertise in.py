# Finding the best market to advertise in!!!
#
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
%matplotlib inline

pd.set_option("display.max_columns", 150)

file_loc = 'D:/Dataquest/Dataquest 2022 Learning/Datasets/'
file_name = '2017-fCC-New-Coders-Survey-Data.csv'
df = pd.read_csv(file_loc + file_name, low_memory=0, encoding='unicode_escape')
# Let inspect the survey dataset by printing the first 5 rows of data and the shape of the data
df.head(5)
df.shape


# %% this is a markdown
