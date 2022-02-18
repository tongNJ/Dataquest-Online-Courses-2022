# Finding the best market to advertise in!!!
#
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns", 150)

file_loc = 'D:/Dataquest/Dataquest 2022 Learning/Datasets/'
file_name = '2017-fCC-New-Coders-Survey-Data.csv'
df = pd.read_csv(file_loc + file_name, low_memory=0, encoding='unicode_escape')
# Let inspect the survey dataset by printing the first 5 rows of data and the shape of the data
df.head(5)
df.shape

# Q1. Are people interested in only one subject or they canbe interested in more than one subject?
df['JobRoleInterest'] = df['JobRoleInterest'].replace(to_replace='nan', value=np.nan)
job_role = df['JobRoleInterest'].dropna()
job_role.dropna()
job_role = job_role.str.split(pat=',', expand=True)
job_role.head(20)
job_role.shape


job_interest_count = job_role.count(axis=1).value_counts(normalize=True)*100
job_interest_count
# Try to plot a waterfall chart
# Create the steps we use to show the changes
job_interest_count.cumsum()
blank = job_interest_count.cumsum().shift(1).fillna(0)
total = job_interest_count.sum()
job_interest_count.loc['Total'] = total
blank.loc['Total'] = total

step = blank.reset_index(drop=True).repeat(3).shift(-1)
step[1::3] = np.nan

blank.loc['Total'] = 0


job_plot = job_interest_count.plot(kind='bar', stacked=True, bottom=blank, figsize=(15, 8))
job_plot.plot(step.index, step.values)
job_plot.set_ylabel('Percentage', fontsize=15)
job_plot.set_xlabel('Number of Subjects Interested', fontsize=15)
job_plot.yaxis.set_major_formatter(mtick.PercentFormatter())

max = job_interest_count.max()
neg_offset = max/25
pos_offset = max/50
plot_offset = int(max/15)
plot_offset

loop = 0
y_height = job_interest_count.cumsum().shift(1).fillna(0)
y_height.reset_index(drop=True, inplace=True)

for index, row in job_interest_count.iteritems():
    if row == total:
        y = y_height[loop]
    else:
        y = y_height[loop] + row

    if row > 0:
        y += pos_offset
    else:
        y -= neg_offset

    job_plot.annotate("{:.1%}".format(row/100), (loop, y), ha='center')
    loop += 1

# After analysing JobRoleInterest column, we found that 32% of the total survey attendee knows which specific job direction (s)he would like to persue

# Q2 - the focus of our courses is on web and mobile development. How many people are interested in at least one of these two subjects??
# There are thousands of different job descriptions in 'JobRoleInterest' columns, and a lot of people are interested in more than one subjects. Let's firstly understand the
# 'JobRoleInterest' columns and see how many job descriptions there are. The we see how many are web and mobile replated subjects.
# job_list = []
# job_role = df['JobRoleInterest'].astype('str').copy()
# job_role = job_role.str.replace(pat=' ,', repl=',').str.replace(
#     pat=', ', repl=',').str.replace(pat=',  ', repl=',')
# job_role = job_role.str.lstrip().str.rstrip()
#
# for i in job_role:
#     if i == 'nan':
#         pass
#     else:
#         job_list += i.split(',')
#
# job_list_unique = []
# for i in job_list:
#     if i in job_list_unique:
#         pass
#     else:
#         job_list_unique.append(i)
#
# job_list_unique

# Lets filter 'JobRoleInterest' columsn tht contains a partile string e.g. 'web','mobile'
keep = ['Web', 'web', 'WEB', 'Mobile', 'mobile', 'MOBILE']
df['JobRoleInterest'] = df['JobRoleInterest'].astype('str')
df_core = df[df['JobRoleInterest'].str.contains('|'.join(keep))].copy()
df_core['JobRoleInterest']

df_core['JobRoleInterest'][2].count(',')

df_core['JobInterest_count'] = df_core['JobRoleInterest'].apply(lambda x: x.count(',')+1)
df_core[['JobRoleInterest', 'JobInterest_count']]
df_core['JobInterest_count'].value_counts(normalize=True)


web_string = ['web', 'Web', 'WEB']
df_core['web_interest'] = df_core['JobRoleInterest'].apply(
    lambda x: 1 if any(y in x for y in web_string) else 0)
df_core[['JobRoleInterest', 'web_interest']].head(20)

# Within 6035 samples, 95% are interested in web related subjects.
df_core['web_interest'].value_counts(normalize=True)*100

mobile_string = ['mobile', 'Mobile', 'MOBILE']
df_core['mobile_interest'] = df_core['JobRoleInterest'].apply(
    lambda x: 1 if any(y in x for y in mobile_string) else 0)
df_core[['JobRoleInterest', 'mobile_interest']].head(20)
# Within 5035 samples, 61% are interested in mobile related subjects.
df_core['mobile_interest'].value_counts(normalize=True)*100

df_core[['JobRoleInterest', 'web_interest', 'mobile_interest']]
df_core['web_mobile_interest'] = df_core['web_interest'] + df_core['mobile_interest']
df_core[['JobRoleInterest', 'web_interest', 'mobile_interest', 'web_mobile_interest']]


def web_mobile_count(x):
    if x['web_interest'] == 1 and x['mobile_interest'] == 0:
        a = 'Web Only'
    elif x['web_interest'] == 0 and x['mobile_interest'] == 1:
        a = 'Mobile Only'
    else:
        a = 'Web and Mobile'
    return a


df_core['web_mobile_interest'] = df_core.apply(web_mobile_count, axis=1)

df_core['web_mobile_interest'].value_counts()

# Now we have a clear picture, out of our 18175 samples, 6992 survey attendees have indicated their interested job type,
# and out of 6992 samples, 6035 attendees experssed their interest in at least one of Web Development or Mobile Development.
job_role.shape
total_job_reply = len(job_role)
total_job_web_mobile = len(df_core)
plt.bar(x=['Web and Mobile Development', 'Other subject'], height=[
        total_job_web_mobile, total_job_reply-total_job_web_mobile])
# Morjority of the attendees are interested in web and mobile development related subjects

df_core['web_mobile_interest'].value_counts().plot.barh()
plt.title('Web and Mobile Subjects')
plt.show()

# With in 6035 samples, 66% are only interested in
df_core['web_mobile_interest'].value_counts(normalize=True)*100


# Q3 finding which are the two best markets to do advertisment
df_adv = df.copy()
df_adv['JobRoleInterest'] = df_adv['JobRoleInterest'].replace("nan", np.nan)
df_adv.dropna(subset=['JobRoleInterest'], inplace=True)
print(df_adv.shape)
df_adv['JobRoleInterest'].head()

df_adv[['CountryLive', 'JobRoleInterest']].head(20)

df_adv['CountryLive'].value_counts().sort_values(ascending=False)
print(df_adv['CountryLive'].value_counts().sum())
df_adv['CountryLive'].value_counts(normalize=True)*100

df_adv['MoneyForLearning'] == 0
df_adv.loc[df_adv['MoneyForLearning'].isnull(), ['MoneyForLearning', 'MonthsProgramming']]

# It looks like that us and India are the two markets for advertisment.
country = ['United States of America', 'India', 'United Kingdom', 'Canada']
# df_adv.dropna(subset=['CountryLive'], inplace=True)
df_adv.dropna(subset=['CountryLive'], inplace=True)
df_adv.shape
# df_focus = df_adv[df_adv['CountryLive'].str.contains('|'.join(country))].copy()
# df_focus['MoneyForLearning']
df_adv['MonthsProgramming'].replace(0, 1, inplace=True)

df_adv['MonthSpend'] = df_adv['MoneyForLearning']/df_adv['MonthsProgramming']
df_adv.loc[df_adv['MonthSpend'].isnull(), ['MoneyForLearning', 'MonthsProgramming', 'MonthSpend']]
df_focus = df_adv.loc[df_adv['MonthSpend'].notnull(), :].copy()
df_focus['MonthSpend'].value_counts(dropna=False).sort_index()

# df_focus2 = df_focus.loc[(df_focus['MonthSpend']>0),:].copy()
df_focus2 = df_focus.copy()
df_focus2['MonthSpend'].value_counts(dropna=False).sort_index()

df_focus2['CountryLive'].value_counts(dropna=False)

df_focus2['MonthSpend'].value_counts(dropna=False)

df_focus2['MonthSpend'].isnull()
avg_spend_country = df_focus2[['CountryLive', 'MonthSpend']].groupby(by='CountryLive').mean()
avg_spend_country.loc[country]
avg_spend_country.plot.bar()

df_focus2[['CountryLive', 'MonthSpend']].groupby(by='CountryLive').median()


df_focus2.loc[df_focus2['CountryLive'] == 'United States of America', 'MonthSpend'].plot.box()
