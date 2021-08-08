#!/usr/bin/env python
# coding: utf-8

# # Hotel Data Analysis - Prediction of Cancellation
# 
# ## Workshop on Data Science

# In[1]:


#todo add imge 


# In[2]:


#todo add Explanation of the problem
# Where does the data come from, what are the contracts, what do we want to show, what will we think of an assessment index, etc.


# In[3]:


# Table of Contents


# In[ ]:





# ### 0.1 install libraries

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### 0. LOAD DATA

# In[5]:


# !pip install pandas
# !pip install seaborn
# !pip install sklearn
hotel_data = pd.read_csv("hotel_bookings.csv")


# 

# In[6]:


sns.set_style("whitegrid")


# In[7]:


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


# ### 0.3 Reading the Datase

# # 1. Exploration (EDA) & Visualization
# 

# ## Data Info :

# ### Size of the Dataset

# In[8]:


hotel_data.shape


# ### Columns & Data types

# In[9]:


hotel_data.head(4)


# In[10]:


# data type distribution 
hotel_data.info()


# #### There are 32 columns.
# * 12 Categorical
# * 20 Numerical
# 
# #### There are 4 columns with the missing values-
# * country
# * agent
# * company
# * children

# ##  Distribution :

# ### Data Statistics

# In[11]:


hotel_data.describe()


# * Most of the columns are full - according to the count line
# * Avg. lead time is 104 days, around 3.5 months.
# * Each booking has on an average 1.8 adults and 0.1 children.
# * Only 3% of the guests are repeated.
# * Median lead time is 69 days.

# ###  Class distribution

# In[12]:


eda_data = hotel_data.copy()
eda_data['is_canceled'] = eda_data.is_canceled.replace([1,0],["Cancelled","Not Cancelled"])
eda_data['is_repeated_guest'] = hotel_data.is_repeated_guest.replace([1,0],["Repeated","Not Repeated"])


sns.countplot(x='is_canceled', data=eda_data)
plt.title('Canceled Distributions', fontsize=14)


# In[13]:


hotel_data['is_canceled'].value_counts()/hotel_data.shape[0]*100


# #### Canceled Distribution :
# * Not Canceled: 62.96%
# * Canceled: 37.04%

# ## Introducing the various features :
# 

# In[14]:




numeric_variables_normal = ['lead_time','arrival_date_week_number','total_of_special_requests']

numeric_variables_normal2 = ['arrival_date_day_of_month', 'arrival_date_year','stays_in_weekend_nights',
                             'stays_in_week_nights','babies', 'booking_changes', 'company', 'adr']


numeric_variables_log1 = ['adults','children','previous_cancellations']
numeric_variables_log2 = ['previous_bookings_not_canceled','days_in_waiting_list','required_car_parking_spaces']


# In[15]:



#      reservation_status_date
#  30  reservation_status              


# ### Categorical Variables :

# In[16]:


categorical_variables = ['country', 'market_segment', 'agent','arrival_date_month', 'meal', 'reserved_room_type', 'assigned_room_type' ]
for i in categorical_variables:
    print(("{} : {} \n").format(i,hotel_data[i].nunique()))


# In[17]:


# plt.figure(figsize=(18,6))
# country_booking = hotel_data['country'].value_counts(normalize=True).rename_axis('country').reset_index(name='Percentage')
# sns.barplot(x='country', y='Percentage', data=country_booking.head(10))
# plt.title('Country of Customers')
# plt.show()


# * PRT or Portugal has the most booking demand based on the data (more than 60%). It is pretty obvious because if we trace to the publication page, the description tells us that the data source locations are from hotels located in Portugal.

# In[18]:


#generate a figures grid:
fig, axes = plt.subplots(2,3,figsize=(22,12))
fig.subplots_adjust(hspace=0.5)

#we will create a histogram for each categorical attribute
n=len(categorical_variables[1:])
num_rows = 3
max_bars = 8

for i,variable in enumerate(categorical_variables[1:]):
    #calculate the current place on the grid
    r=int(i/num_rows)
    c=i%num_rows
    
    #create the "value counts" for the first <max_bars> categories:
    u=min(hotel_data[variable].nunique(),max_bars)
    vc = hotel_data[variable].value_counts()[:u]
    
    # plot a bar chart using Pandas
    vc.plot(kind='bar',ax=axes[r,c],title=variable , color="rbgkm"[i%5])


# * reserved_room_type and assigned_room_type similar, We will examine later the option to unify or download one of them
# * Agent 9 is the most popular
# * We will present the orders by months in a more orderly manner below
# * The most common meal is BB (Bed & Breakfast)

# #### Binary Variables :

# In[19]:


binary_variables = ['hotel', 'is_repeated_guest','is_canceled', 'distribution_channel', 'deposit_type', 'customer_type']


# In[20]:


plt.rcParams.update({'font.size': 10})
#initialize a Matplotlib figures grid
fig, axes = plt.subplots(2,3,figsize=(18,8))
                         
fig.subplots_adjust(hspace=0.5)

#we will create a histogram for each categorical attribute
n=len(binary_variables)
num_rows = 3
                    
#generate a histogram using Pandas, for each numeric variable
for i, var in enumerate(binary_variables):
    r=int(i/num_rows)
    c=i%num_rows         
    eda_data[var].value_counts().plot.pie(ax=axes[r,c] ,autopct="%.2f%%")
    


# * Most hotels are City hotel: 66% 
# * The vast majority of customers are not repeat visitors
# * In the most of the hotel no policy of diposit
# * Most of thr custumer are transient (when the booking is not part of a group or contract)
# * The most poplar Distribution Channel is “TA” means “Travel Agents” and “TO” means “Tour Operators”

# ### Numeric Variables :

# In[21]:


numeric_variables_normal = ['lead_time','arrival_date_week_number','total_of_special_requests']

numeric_variables_normal2 = ['arrival_date_day_of_month', 'arrival_date_year','stays_in_weekend_nights',
                             'stays_in_week_nights','babies', 'booking_changes', 'company', 'adr']


# In[22]:


# Show numeric_variables
#initialize a Matplotlib figures grid
fig, axes = plt.subplots(1, len(numeric_variables_normal),figsize=(18,3))

#generate a histogram using Pandas, for each numeric variable
for ind,var in enumerate(numeric_variables_normal):
    hotel_data[var].hist(ax=axes[ind],edgecolor='black' ,color="rbgkm"[ind%5])
    axes[ind].set_title(var)


# * Week 30 is the most popular week for August bookings (holiday)
# * Earlier lead time is more common
# * Most visitors do not make special requests

# #### Distribution of lead time:

# In[23]:


# sns.kdeplot(hotel_data["lead_time"], kernel='epa')


# * Avg. lead time is 104 days, around 3.5 months.
# * Median lead time is 69 days.
# * The decrease in lead time can be seen as time increases
# 

# In[24]:


fig, axes = plt.subplots(2,4,figsize=(18,8))
fig.subplots_adjust(hspace=0.5)

#we will create a histogram for each categorical attribute
n=len(numeric_variables_normal2)
num_rows = 4

for i,var in enumerate(numeric_variables_normal2):
    #calculate the current place on the grid
    r=int(i/num_rows)
    c= i%num_rows
    
    
    hotel_data[var].hist(ax=axes[r,c] ,edgecolor='black' ,color="rbgkm"[i%5]).set_title(var)

    


# * Arrival time is kind of uniform distributed, the most popular is arrival at the end of the month
# * At 2016 was most of the arrival
# * There are 236 different companies (id) where some of the data in them is null
# 
# #### Some of the plots are not informative so we will show them more plots:

# In[25]:


fig, axs = plt.subplots(1,3, figsize=(22,3))

sns.boxplot(hotel_data.stays_in_weekend_nights, ax=axs[0])
axs[0].set_title("Stays in weekend - box plot")

sns.boxplot(hotel_data.stays_in_week_nights, ax=axs[1])
axs[1].set_title("stays in week nights - box plot")


country_booking = hotel_data['stays_in_week_nights'].value_counts(normalize=True).rename_axis('stays_in_week_nights').reset_index(name='Percentage')
sns.barplot(ax=axs[2], x='stays_in_week_nights', y='Percentage', data=country_booking.head(10))
plt.title('stays_in_week_nights')


# * There are extreme values - people who have booked for more than 5 weeks (over a month)
# * The avarge pf stay in weekend is 1.19 day and weekday is 3.13
# * More than 25 percent order 2 days a week

# In[26]:


fig, axs = plt.subplots(1,3, figsize=(22,3))


country_booking = hotel_data['babies'].value_counts(normalize=True).rename_axis('babies').reset_index(name='Percentage')
sns.barplot(ax=axs[0], x='babies', y='Percentage', data=country_booking.head(5))
axs[0].set_title('babies (top 5)')

cg = hotel_data[hotel_data["babies"] > 0]
country_booking = cg['babies'].value_counts(normalize=True).rename_axis('babies').reset_index(name='Percentage')
sns.barplot(ax=axs[1], x='babies', y='Percentage', data=country_booking.head(5))
axs[1].set_title('babies more than 1 (top 5)')


country_booking = hotel_data['booking_changes'].value_counts(normalize=True).rename_axis('booking_changes').reset_index(name='Percentage')
sns.barplot(ax=axs[2], x='booking_changes', y='Percentage', data=country_booking.head(5))
plt.title('booking_changes')


# * The vast majority of orders are with zero babys
# * Those who did book with a baby usually booked for a single baby
# * Most people do not change their order

# In[27]:


# fig, axs = plt.subplots(1,2, figsize=(12,3))

# sns.kdeplot(hotel_data["adr"], kernel='epa', ax=axs[0])
# axs[0].set_title("Adr - kdeplot")

# # days in waiting list not 0.. (boxplot)

# cg = hotel_data[hotel_data["adr"] < 1000]
# sns.boxplot(cg["adr"], ax=axs[1])
# axs[1].set_title("Adr - box plot")


# #### ADR - Average Daily Rate
#     Calculated by dividing the sum of all lodging transactions by the total number of staying nights
# * Can see an average in the 95 range and most of the data is between 90 and 115
# * There is a some data that is considered extreme (over 210)

# In[28]:


numeric_variables_log1 = ['adults','children','previous_cancellations']
numeric_variables_log2 = ['previous_bookings_not_canceled','days_in_waiting_list','required_car_parking_spaces']
# Const
bins = []
for x in range(-1,5400,1):
    bins.append(x+0.5)


# In[29]:


# Show numeric_variables: 
#initialize a Matplotlib figures grid
fig, axes = plt.subplots(1, len(numeric_variables_log1),figsize=(24,3))


#generate a histogram using Pandas, for each numeric variable
for ind,var in enumerate(numeric_variables_log1):
    max_value = int(hotel_data[var].max())
    slice_object = slice(max_value)
    hotel_data[var].hist(ax=axes[ind],bins=bins[slice_object], edgecolor='black', log=True)
    axes[ind].set_title(var)


# * Most bookings are for 1 adult or more (2 is must commen)
# * Most customers do not have children (or at least did not include them in the order)
# * Most people have not previously canceled an order before the current orde
# 
# #### Can see that there are slight noises in the data : 
# * There are data showing over 20 guests and even 50
# * Or order with 10 babies (unlikely)

# In[30]:


# Show numeric_variables:
#initialize a Matplotlib figures grid
fig, axes = plt.subplots(1, len(numeric_variables_log2),figsize=(24,3))

#generate a histogram using Pandas, for each numeric variable
for ind,var in enumerate(numeric_variables_log2):
    max_value = int(hotel_data[var].max())
    slice_object = slice(max_value)
    hotel_data[var].hist(ax=axes[ind],bins=bins[slice_object], edgecolor='black', log=True)
    axes[ind].set_title(var)


# * It can be seen that according to previous orders most customers do not cancel orders
# * Most people do not require parking, and those who do, ask for one

# In[31]:


# days in waiting list not 0.. (boxplot)
fig, axs = plt.subplots(1,1, figsize=(6,4))

cg = hotel_data[hotel_data["days_in_waiting_list"] > 0]
sns.boxplot(cg["days_in_waiting_list"])


# * The average time to be on the waiting list is 0.5
# * The vast majority of straight orderers are happy and not waiting on the waiting list
#   <br>But those who waited usually waited about 50 days
# * Most of the waiters are in the range of 40 to 90
# * There are extreme values that have waited over 170 days (half a year or more)

# ## Comparative Visualizations To Cancellation :

# ### Market segment vs Cancellations

# In[32]:


_, ax = plt.subplots( nrows = 2, ncols = 1, figsize = (12,8))
sns.countplot(x = 'market_segment', data = hotel_data, ax = ax[0])
sns.countplot(x = 'market_segment', data = hotel_data, hue = 'is_canceled', ax = ax[1])
plt.show()


# * It can be seen that the order of the amount of orders in market segment does change in cancellations filter:
# <br> **from -** 
# <br> &emsp; Online TA -> Offline TA -> Groups -> Direct -> Corporate
# <br> **To -**
# <br> &emsp; Online TA -> **Groups** -> **Offline TA** -> Direct -> Corporate
# 
# #### It is possible that order in a group increased the chances of cancellations
# 

# ### Date vs Cancellations

# #### Arrival date month

# In[33]:


# # Number of Canceled Each Month

# # We can simply use a countplot as we sre visualising categorical data
# plt.figure(figsize=(20,5))

# # data we will use in a list
# l1 = ['is_canceled','arrival_date_month']

# # plotting
# sns.countplot(data = hotel_data[l1],x= "arrival_date_month",hue="is_canceled",order=["January","February","March","April","May","June",
#                                                                               "July","August","September","October","November","December"]).set_title(
# 'Illustration of Number of Canceled Each Month')
# plt.xlabel('Month')
# plt.ylabel('Count')


# #### Arrival year

# In[34]:


hotel_data.groupby(['arrival_date_year'])['is_canceled'].mean()


# #### Arrival week number

# In[35]:


fig, axs = plt.subplots(2,1, figsize=(16,12))


# We can simply use a countplot as we sre visualising categorical data
# plt.figure(figsize=(20,5))

# data we will use in a list
l1 = ['is_canceled','arrival_date_week_number']

# plotting
sns.countplot(ax = axs[0], data = hotel_data[l1],x= "arrival_date_week_number").set_title(
'Illustration of Canceled Each week')

# plotting
sns.countplot(ax = axs[1], data = hotel_data[l1],x= "arrival_date_week_number",hue="is_canceled").set_title(
'Illustration of Canceled Each week')


# * Can see similarities between the graphs
# *There are weeks when the cancellation ratio changes slightly but not something extreme

# #### Summary  date vs cancellations:
# * The cancellation rate is quite consistently high during april to october having its peak at august.
# * There is no direct effect between the month and the amount of cancellations
# * This year does not affect the cancellation at all around 36%
# *There are weeks when the cancellation ratio changes slightly but not something extreme
# 
# 

# ### Deposit  type vs Cancellations

# In[36]:


_, ax = plt.subplots( nrows = 1, ncols = 2, figsize = (18,4))
sns.countplot(x = 'deposit_type', data = hotel_data, hue = 'hotel', ax = ax[0])
sns.countplot(x = 'deposit_type', data = hotel_data, hue = 'is_canceled', ax = ax[1])
plt.show()


# * Deposit type has 3 categories - No Deposit, refundable, Non Refund
# * Either customers have opted for no deposit or non refundable deposits.
# * Maybe refundable deposit type is not offered by the hotels.
# * All of the non refund bookings have been cancelled in our dataset. That might prove important feature based on how many such   bookings are part of cancelled bookings.
# * No hotel has refundable deposit type
# 
# * In city hotel is more common Non Refund deposit policy 

# ### Hotel type vs Cancellations

# In[37]:


# Let's look into how much of bookings were cancelled in each type of hotel
fig, axs = plt.subplots(1,2, figsize=(16,3))

sns.countplot(x = 'hotel', data = hotel_data, ax = axs[0], order = hotel_data['hotel'].value_counts().index).set_title('city and resort hotel')

lst1 = ['is_canceled', 'hotel']
type_of_hotel_canceled = eda_data[lst1]
canceled_hotel = type_of_hotel_canceled[type_of_hotel_canceled['is_canceled'] == 'Cancelled'].groupby(['hotel']).size().reset_index(name = 'count')
canceled_hotel
sns.barplot(data = canceled_hotel, x = 'hotel', y = 'count', ax=axs[1]).set_title('Graph showing cancellation rates in city and resort hotel')


# In[38]:


hotel_data.groupby(['hotel'])['is_canceled'].mean()


# * City hotel has high Cancellation rate than Resort Hotel.
# * Around 27% for resort hotel and greater than 40 % for city hotel.
# 
# #### There seems to be a connection between the type of hotel and the chance of cancellation

# ### Customer profile vs Cancellations

# In[39]:


# fig, axs = plt.subplots(1,3, figsize=(22,3))


# # We will just look at number of children that canceled booking.
# sns.countplot(ax=axs[0], data=hotel_data,x='children',hue='is_canceled').set_title("Illustration of number of children canceling booking")

# # We will just look at number of babies that canceled booking.
# sns.countplot(ax=axs[1],data=hotel_data,x='babies',hue='is_canceled').set_title("Illustration of number of babies canceling booking")


# sns.countplot(ax=axs[2],data=hotel_data,x='adults',hue='is_canceled').set_title("Illustration of number of babies canceling booking")


# * There does not appear to be a direct link between the customer profile and the cancellations.
# * We will later choose to try and convert these features to a single feature (After cleaning and arranging)

# ### Customer type vs Cancellations

# In[40]:


_, ax = plt.subplots( nrows = 1, ncols = 2, figsize = (18,5))
sns.countplot(x = 'customer_type', data = hotel_data, ax = ax[0])
sns.countplot(x = 'customer_type', data = hotel_data, hue = 'is_canceled', ax = ax[1])
plt.show()


# * Same order for customer of bookings and cancellation rate

# ### Reservation status vs Cancellations

# In[41]:


# _, ax = plt.subplots( nrows = 1, ncols = 2, figsize = (18,5))
# sns.countplot(x = 'reservation_status', data = hotel_data, ax = ax[0])
# sns.countplot(x = 'reservation_status', data = hotel_data, hue = 'is_canceled', ax = ax[1])
# plt.show()


# * A strong connection can be seen between the two features
# * **Because this is the feature we predict we will delete it at the cleaning stage**

# ## Clean The Data ( part 1 ) :

# In[ ]:





# In[42]:




# rfc=RandomForestClassifier()
# rfc.fit(x_train,y_train)
# RandomForestClassifier()
# y_pred=rfc.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# conf =print(confusion_matrix(y_test, y_pred))
# clf =print(classification_report(y_test, y_pred))
# score=accuracy_score(y_test,y_pred)
# print("Random Forest score: ",score)


# ### Replace missing value

# In[43]:


clean_data = hotel_data.copy()
print(clean_data.shape)

clean_data.fillna({"children": 0}, inplace=True)

# missing countries can be labeled unknown
clean_data.fillna({"country": "Unknown"}, inplace=True)

# missing agent ID can be zero, presuming the booking was made privately
clean_data.fillna({"agent": 0}, inplace=True)

# missing company ID can be zero (for the same reason as agent ID)
clean_data.fillna({"company": 0}, inplace=True)


# * can assume that null is represents no children
# * missing countries can be labeled unknown
# * missing agent ID can be zero, presuming the booking was made privately
# * missing company ID can be zero (for the same reason as agent ID)
# 

# In[44]:


# clean_data = pd.get_dummies(clean_data)
# print(clean_data.shape)
# clean_data.head()


# In[45]:


clean_data.isnull().sum()/clean_data.shape[0]*100


# delet  Reservation status...

# ## Correlation Heat Map of features

# In[46]:


corr_matrix = clean_data.corr(method='spearman')
fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(clean_data.corr(method='spearman'),annot=True,linewidths=.5)


# In[47]:


corr_matrix = clean_data.corr()
corr_matrix["is_canceled"].sort_values(ascending=False)


# #### At this stage, we will examine only the numerical features (later we will convert additional features)
# 
# #### The must strong connections are between cancelesion and this features:
# * lead_time 
# * total_of_special_requests
# * required_car_parking_spaces
# * previous_cancellations

# ## Depth analysis of the strong bond (nunrical feature) :

# ### Lead time - key feature

# In[48]:


# hist plot of lead time
# kde = kernel density estimation (displays distribution function, density curve)
# shows the distribution and highest concentration points
# plt.figure(figsize=(10,5))
# lead_time = clean_data['lead_time']
# lead_time = pd.DataFrame(sorted(lead_time, reverse = True), columns = ['Lead'])
# sns.histplot(lead_time, kde=True)
# plt.title("Lead Time", size=20)
# plt.xlabel("lead time days", size=15)
# plt.tight_layout()
# plt.show()


# In[49]:


lead_time_1 = clean_data[clean_data["lead_time"] < 50]
lead_time_2 = clean_data[clean_data["lead_time"] < 100]
lead_time_3 = clean_data[clean_data["lead_time"] < 150]
lead_time_4 = clean_data[clean_data["lead_time"] < 200]
lead_time_5 = clean_data[(clean_data["lead_time"] >= 200) & (clean_data["lead_time"] < 365)]
lead_time_6 = clean_data[clean_data["lead_time"] >= 365]
# calculates cancellations according to lead time groups
lead_cancel_1 = lead_time_1["is_canceled"].value_counts()
lead_cancel_2 = lead_time_2["is_canceled"].value_counts()
lead_cancel_3 = lead_time_3["is_canceled"].value_counts()
lead_cancel_4 = lead_time_4["is_canceled"].value_counts()
lead_cancel_5 = lead_time_5["is_canceled"].value_counts()
lead_cancel_6 = lead_time_6["is_canceled"].value_counts()


# In[50]:


# total count of lead time according to cancellation
total_lead_days_cancel = pd.DataFrame(data=[lead_cancel_1,lead_cancel_2,lead_cancel_3,lead_cancel_4,lead_cancel_5,lead_cancel_6],
                                      index=["[0,50) days","[50,100) days","[100,150) days","[150,200) days","200,365) days","[365,max) days"])

# pie plot for each lead time group
fig, ax = plt.subplots(2,3, figsize=(15,6))
ax[0,0].pie(np.array([total_lead_days_cancel[0][0], total_lead_days_cancel[1][0]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[0,0].set_title("lead_time [0,50) days", size=15)
ax[0,1].pie(np.array([total_lead_days_cancel[0][1], total_lead_days_cancel[1][1]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[0,1].set_title("lead_time [50,100) days", size=15)
ax[0,2].pie(np.array([total_lead_days_cancel[0][2], total_lead_days_cancel[1][2]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[0,2].set_title("lead_time [100,150) days", size=15)


ax[1,0].pie(np.array([total_lead_days_cancel[0][3], total_lead_days_cancel[1][3]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[1,0].set_title("lead_time [150,200) days", size=15)

ax[1,1].pie(np.array([total_lead_days_cancel[0][4], total_lead_days_cancel[1][4]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[1,1].set_title("lead_time [200,356) days", size=15)

ax[1,2].pie(np.array([total_lead_days_cancel[0][5], total_lead_days_cancel[1][5]]),
          labels=["not_canceled", "canceled"], autopct='%1.1f%%', startangle=90,
          colors=['forestgreen', 'firebrick'])
ax[1,2].set_title("lead_time [356,max) days", size=15)

plt.tight_layout()
plt.show()


# #### It can be clearly seen, as the lead time increases the chance of cancellation increases :
# * And that does make sense, since there is more time to cancel, and the chances of change are greater

# ### Special Requests

# In[51]:


# plot special requests according to cancellations
# plt.figure(figsize=(10,5))
# sns.countplot(x=clean_data["total_of_special_requests"], hue=clean_data["is_canceled"])
# plt.title("Special Requests", size=20)
# plt.xlabel("Number of Special Requests", size=15)
# plt.legend(["not canceled", "canceled"])
# plt.tight_layout()
# plt.show()

# Nearly half of the bookings without special requests are canceled.


# #### There is a strong connection between the number of requests and the cancellations
# * The more requests there are the chance of cancellation decreases significantly
# * One request is enough to greatly reduce the chance of cancellation

# ## A combination of the two features

# In[52]:


var_a = 'lead_time'
var_b = 'total_of_special_requests'

canceled = clean_data[clean_data['is_canceled'] == 1]
not_canceled = clean_data[clean_data['is_canceled'] == 0]


# In[53]:


sns.set(color_codes=True)
fig,ax=plt.subplots(figsize=(16,8))

sns.regplot(var_a, var_b, canceled,ax=ax, 
            scatter_kws={"marker": ".", "color": "blue"},
            line_kws = {"linewidth": "1", "color": "blue"},
            order = 3,
            label = 'canceled')
sns.regplot(var_a, var_b, not_canceled,ax=ax, 
            scatter_kws={"marker": ".", "color": "orange"},
            line_kws = {"linewidth": "1", "color": "orange"},
            order = 3,
            label = 'not canceled')

fig.legend(loc="lower right")
fig.suptitle(f"Scatter plot of {var_a} and {var_b}")


# ### The following plot shows the two points presented earlier
# * Relationship between the number of requests and cancellations
# * Relationship between order time and cancellations
# 
# ### In the plot it can be seen that looking at the two features together allows for understanding (partial prediction) at a higher probability.
# * The sheer majority of the blue dots in the lower right quarter of the graph

# ## Feature Engineering & Clean The Data ( part 2 ) :

# ### Naive approach to converting categories to numbers

# In[54]:


hot_data = clean_data.copy()
hot_data.shape


# ### Categories to Numbers :

# In[55]:


# def setDictionary(dictionary,unique_data_arr): 
#     for i in range (len(unique_data_arr)):
#         dictionary[unique_data_arr[i]] = i
#     return dictionary


# In[56]:


# num_data = clean_data.copy()

# # hotel to bool
# num_data['hotel']= num_data['hotel'].replace(["Resort Hotel","City Hotel"],[1,0])

# # arrival_date_month to int 
# month_dict = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11} 
# num_data['arrival_date_month']= num_data['arrival_date_month'].map(month_dict)

# # meal to int 
# meal_dict = {'BB':1, 'FB':2 ,'HB':3, 'SC':4 ,'Undefined':0}
# num_data['meal'] = num_data['meal'].map(meal_dict)


# # countery to int 
# country_dict ={}
# country_arr = num_data['country'].unique()
# country_dict = setDictionary(country_dict, num_data['country'].unique())
# num_data['country'] = num_data['country'].map(country_dict)

# # market_segment to int 
# market_segment_dict ={}
# market_segment_dict = setDictionary(market_segment_dict,num_data['market_segment'].unique())
# num_data['market_segment'] = num_data['market_segment'].map(market_segment_dict) 

# # distribution_channel to int
# distribution_channel_dict = {}
# distribution_channel_dict = setDictionary(distribution_channel_dict,num_data['distribution_channel'].unique())
# num_data['distribution_channel'] = num_data['distribution_channel'].map(distribution_channel_dict) 

# # reserved_room_type to int
# reserved_room_type_dict = {}
# reserved_room_type_dict = setDictionary(reserved_room_type_dict,num_data['reserved_room_type'].unique())
# num_data['reserved_room_type'] = num_data['reserved_room_type'].map(reserved_room_type_dict)

# # assigned_room_type_dict TO INT 
# assigned_room_type_dict = {}
# assigned_room_type_dict = setDictionary(assigned_room_type_dict, num_data['assigned_room_type'].unique())
# num_data['assigned_room_type'] = num_data['assigned_room_type'].map(assigned_room_type_dict)

# # deposit_typ TO INT 
# deposit_type_dict = {}
# deposit_type_dict = setDictionary(deposit_type_dict,num_data['deposit_type'].unique())
# num_data['deposit_type'] = num_data['deposit_type'].map(deposit_type_dict)

# # customer_type TO INT 
# customer_type_dict = {}
# customer_type_dict = setDictionary(customer_type_dict, num_data['customer_type'].unique())
# num_data['customer_type'] = num_data['customer_type'].map(customer_type_dict)

# # reservation_status TO INT 
# reservation_status_dict = {}
# reservation_status_dict = setDictionary(reservation_status_dict,num_data['reservation_status'].unique())
# num_data['reservation_status'] = num_data['reservation_status'].map(reservation_status_dict)

# # reservation_status_date TO INT 
# reservation_status_date_dict ={}
# reservation_status_date_dict =setDictionary(reservation_status_date_dict, num_data['reservation_status_date'].unique())
# num_data['reservation_status_date'] = num_data['reservation_status_date'].map(reservation_status_date_dict)

# num_data.shape


# ### Create numerical data:
# * From 32 colume to 32 
# * The features are now dependent of the order
# 
# ### Preparation of this model is for comparison only<br> (class presentation)

# ### Delete features :
# * We will check if there are strong ties with the features we created

# In[57]:


# corr_matrix = num_data.corr()
# corr_matrix["is_canceled"].sort_values(ascending=False)


# #### High dependence between reservation_status and is_canceled  (0.980601)

# In[58]:


# num_data = num_data.drop('reservation_status_date', axis = 1)
# num_data = num_data.drop('reservation_status', axis = 1)
# print(clean_data.columns)
# clean_data = clean_data.drop('reservation_status_date', axis = 1)
# clean_data = clean_data.drop('reservation_status', axis = 1)


# * Delete reservation_status & reservation_status_date

# TODO Comparative Visualizations To Cancellation : (The categorical features)

# ### Categories to One Hot :

# In[59]:


# hot_data = hot_data.drop('reservation_status_date', axis = 1)
# hot_data = hot_data.drop('reservation_status', axis = 1)


# * Delete reservation_status & reservation_status_date

# In[60]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

hot_data.shape
hot_data = pd.get_dummies(hot_data, prefix='Category_', columns=['hotel','arrival_date_month','meal','country','market_segment',
                                                                'distribution_channel','reserved_room_type','assigned_room_type',
                                                                'deposit_type','customer_type'])


# In[61]:


hot_data.shape


# In[62]:


# print(hot_data.info)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# y = clean_data['is_canceled']
# X = clean_data.drop('is_canceled',axis = 1)
# x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# rfc=RandomForestClassifier()
# rfc.fit(x_train,y_train)
# RandomForestClassifier()
# y_pred=rfc.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# conf =print(confusion_matrix(y_test, y_pred))
# clf =print(classification_report(y_test, y_pred))
# score=accuracy_score(y_test,y_pred)
# print("Random Forest score: ",score)

# print(hot_data.info)


# ### Hot_data :
# * From 20 colume to 259 
# * Delete reservation_status
# * The features are now independent of the order

# ### Create combination data of hot vector and numbers

# In[63]:


clean_data.drop(['hotel','arrival_date_month','meal','country','market_segment','distribution_channel',
                                      'reserved_room_type','reservation_status','assigned_room_type','deposit_type','customer_type','reservation_status_date'],axis=1,inplace=True)
clean_data.columns


# In[64]:


# clean_data['arrival_date_month'] = num_data['arrival_date_month']
# clean_data['hotel'] = num_data['hotel']
# remove_country_data = clean_data.copy()
print(clean_data.info())
# clean_data = clean_data.drop('reservation_status',axis=1)


# In[65]:


# clean_data = pd.get_dummies(clean_data, prefix='Category_', columns=['meal','country','market_segment', 'distribution_channel',
#                                                                      'reserved_room_type','assigned_room_type',
#                                                                     'deposit_type','customer_type','arrival_date_month','hotel'])


# In[66]:


from datetime import datetime as dt
# clean_data  = clean_data.drop('reservation_status_date',axis = 1)
print(clean_data.info())


# ### Create Combination Data (clean data) :
# * Arrival date month - to number (save the order)
# * All ohter is hot vector

# In[67]:


from sklearn.model_selection import train_test_split
y = clean_data['is_canceled']
clean_data.drop(['is_canceled'],axis = 1,inplace=True)
print(clean_data.columns)


# In[68]:


X = clean_data
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[69]:


# corr_matrix = clean_data.corr(method='spearman')
# fig, ax = plt.subplots(figsize=(25,25))
# sns.heatmap(clean_data.corr(method='spearman'),annot=True,linewidths=.5)


# In[70]:


#Random Forest 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, roc_auc_score,auc

rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
RandomForestClassifier()
y_pred=rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
print("Random Forest score: ",score)


# In[71]:


# remove_country_data = remove_country_data.drop('country', axis=1)
# remove_country_data = pd.get_dummies(remove_country_data, prefix='Category_', columns=['meal','market_segment', 'distribution_channel',
#                                                                      'reserved_room_type','assigned_room_type',
#                                                                     'deposit_type','customer_type'])


# ### Created Small Combination data:
# * Arrival date month - to number (save the order)
# * Delete country feature - 177 unique
# * All ohter is hot vector

# ## Shapes of the various data :

# In[72]:


# print(f'clean data (A combination of hot and number) shape {clean_data.shape}')
# print(f'hot vector (hot only) data shape {hot_data.shape}')
# print(f'num data (number only) shape {num_data.shape}')
# print(f'remove_country_data (like the first affter remove country) shape {remove_country_data.shape}')


# * Note that the number of columns depends on the change in the data
# * Hot vector increases the number of columns

# In[73]:


# data = {"clean_data": clean_data, "hot_data": hot_data, "num_data": num_data, "remove_country_data": remove_country_data}


# #  2.  Naive Model :

# ## Imports

# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# ## Activating the simple model
# 

# In[75]:


def train(clean_data, class_w):
    y = clean_data['is_canceled']
    X = clean_data.drop('is_canceled',axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    
    clf = LogisticRegression(random_state=0, class_weight=class_w).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    predicted_probs = clf.predict_proba(X_test)
    
    return({"clf": clf, "y_t": y_test, "y_p": y_pred, "p_p": predicted_probs, "x_train": X_train, "y_train":y_train, 'x_test': X_test})

    


# ### Deatils
# * Logistic Regression model
# * Division 20% to test and 80% train
# * Class_weight : 'weights' Or 'balanced'

# In[76]:


class_w = ['weights', 'balanced']


# In[77]:


# ans = {}
# for key in data:
#     for c_w in class_w:
#         ans[key+'_'+ c_w] = train(data[key], c_w)

# for key, value in ans.items():
#     print(f'{key} accuracy -> {accuracy_score(value["y_t"],value["y_p"])}')


# In[78]:


#TODO delte warninng


# ### Model results :
# * Clean_Data has the best performance (78.4%)
# * Class_weight 'balanced' decreases accuracy Raises the recall
# * Conversion to numbers led to a decrease in performance (confused the model)

# In[79]:


# for key, value in ans.items():
#     print(f' \n  {key} :')

#     values = value["y_t"].value_counts(dropna=False).keys().tolist()
#     counts = value["y_t"].value_counts(dropna=False).tolist()
#     print(f'y_test: {dict(zip(values, counts))}')
    
#     unique, counts = np.unique(value["y_p"], return_counts=True)
#     print(f'y_prid: {dict(zip(unique, counts))}')
    


# In[80]:


#todo show in plot comper (only clean_data_weights & clean_data_balanced  )


# In[81]:


### 'weights' VS 'balanced'
# todo add note


# In[82]:


# clf, y_test, y_pred , predicted_probs, x_train , y_train, x_test = ans['clean_data_weights'].values()
# y_score = predicted_probs[:,1]


# In[83]:


#TODO add croosvalidtion on  clean data weights' and 'balanced'


# # 3. Evaluate results (Naiv)
# 

# In[84]:


from sklearn import metrics
import plotly.figure_factory as ff


# ### Comparison True positive, false positive..

# In[85]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)


# In[86]:


import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()


fig = ff.create_annotated_heatmap(z=mat,
                                  x=["False", "True"], y=["False", "True"], 
                                  showscale=True)
fig.update_layout(font=dict(size=12))
# Add Labels
fig.add_annotation(x=0,y=0, text="True Negative", 
                   yshift=40, showarrow=False, font=dict(color="black",size=24))
fig.add_annotation(x=1,y=0, text="False Positive", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))
fig.add_annotation(x=0,y=1, text="False Negative", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))
fig.add_annotation(x=1,y=1, text="True Positive", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))

fig.update_xaxes(title="Predicted")
fig.update_yaxes(title="Actual", autorange="reversed")


# #### Comparison True positive, false positive and so on:
# Advance data 62 percent without cancellation.
# * So it makes sense- The majority is true negative

# ### Comparison precision, recall,  f1-score

# In[87]:


# for key, value in ans.items():
#     unique, counts = np.unique(value["y_p"], return_counts=True)
#     print(f' \n  {key} :')
    
#     print(metrics.classification_report(value["y_t"], value["y_p"]))


# In[88]:


# from sklearn.metrics import precision_recall_fscore_support as score
# balanced = ans['clean_data_weights']
# w = score(balanced["y_t"], balanced["y_p"])


# In[89]:


# balanced = ans['clean_data_balanced']
# b = score(balanced["y_t"], balanced["y_p"])


# In[90]:


# weights_not_canceled = [w[0][0],w[1][0],w[2][0]]
# balanced_not_canceled = [b[0][0],b[1][0],b[2][0]]

# weights_is_canceled = [w[0][1],w[1][1],w[2][1]]
# balanced_is_canceled = [b[0][1],b[1][1],b[2][1]]
# index = ['Precision ', 'Recall', 'F-score']


# pd1 = pd.DataFrame({'weights': weights_not_canceled,
#                    'balanced': balanced_not_canceled}, index=index)

# pd2 = pd.DataFrame({'weights': weights_is_canceled,
#                    'balanced': balanced_is_canceled}, index=index)

# fig, axes = plt.subplots(ncols=2,figsize=(18,5))
# pd1.plot.bar(ax = axes[0])
# axes[0].set_title("Not Canceled Performence")
# axes[0].set_xlabel("Evaluation")
# axes[0].set_ylabel("Score")
# pd2.plot.bar(ax = axes[1])
# axes[1].set_title("Is Canceled Performence")
# axes[1].set_xlabel("Evaluation")
# axes[1].set_ylabel("Score")


# recall ->As expected no higher recall was canceled
# balanced -> make priecision to be higher 
# 

# In[91]:


def plotPR(precision, recall):
    plt.figure()
    plt.plot(recall, precision, label='PR curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.legend(loc='lower left')
    plt.show()


# In[92]:


# import sklearn
# precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_score, pos_label=1)
# plotPR(precision, recall)


# In[93]:


# Todo add Explanation of the plot


# In[94]:


### ROC Curve


# In[95]:


def plotRoc(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characterist')
    plt.legend(loc="lower right")
    plt.show()


# In[96]:


# auc = sklearn.metrics.roc_auc_score(y_test, y_score)
# fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)
# plotRoc(fpr, tpr, auc)


# ## Understanding the results of the naive model
# 

# In[97]:


import shap
shap.initjs()


# In[98]:


# clf, y_test, y_pred , predicted_probs, x_train , y_train, x_test = ans['clean_data_weights'].values()


# In[99]:


# def predict_fcn(x):
#     return clf.predict_proba(x)[:,1]


# In[100]:


# background_data = shap.maskers.Independent(x_train,  max_samples=100)
# explainer = shap.Explainer(predict_fcn, background_data)


# In[101]:


# shap_values_100 = explainer(x_test[:100])


# In[102]:


# shap.plots.waterfall(shap_values_100[5], max_display=14)


# In[104]:


shap.plots.beeswarm(shap_values_100, max_display=8)


# In[105]:


# def perf_measure(y_actual, y_hat, x_t):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
    

#     for i in range(len(y_hat)): 
# #         if y_actual[i]==y_hat[i]==1:
# #            TP += 1
#         if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
#             FP += 1
#             if FP>100:
#                 break
# #         if y_actual[i]==y_hat[i]==0:
# #            TN += 1
# #         if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
# #             FN += 1
# #             if FN>100:
# #                 break
#         else :
#             x_t.drop(x_t.iloc[i].name, inplace=True)
        

#     return(FP, x_t)


# In[106]:


# x_copy = x_test.copy()
# FP, x_t = perf_measure(y_test.values, y_pred, x_copy )


# In[107]:


# background_data = shap.maskers.Independent(x_train,  max_samples=100)
# explainer = shap.Explainer(predict_fcn, background_data)
# shap_values = explainer(x_t[:100])


# In[108]:


# shap.plots.beeswarm(shap_values, max_display=8)


# ### Similar behavior can be seen both in false negative and in general and therefore, we did not find it appropriate to work on a particular feature

# # 4. The improved model 
# 

# In[109]:


y = hot_data['is_canceled']
hot_data.drop(['is_canceled','reservation_status','reservation_status_date'],axis = 1,inplace=True)


# In[110]:


from sklearn.model_selection import train_test_split
X = hot_data
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[111]:


print(hot_data.select_dtypes(include=['object']))
# print(hot_data.info())


# 

# In[ ]:





# 

# In[112]:


#Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
RandomForestClassifier()
y_pred=rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
print("Random Forest score: ",score)


# In[113]:


import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()


fig = ff.create_annotated_heatmap(z=cm,
                                  x=["False", "True"], y=["False", "True"], 
                                  showscale=True)
fig.update_layout(font=dict(size=12))
# Add Labels
fig.add_annotation(x=0,y=0, text="True Negative", 
                   yshift=40, showarrow=False, font=dict(color="black",size=24))
fig.add_annotation(x=1,y=0, text="False Positive", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))
fig.add_annotation(x=0,y=1, text="False Negative", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))
fig.add_annotation(x=1,y=1, text="True Positive", 
                   yshift=40, showarrow=False, font=dict(color="white",size=24))

fig.update_xaxes(title="Predicted")
fig.update_yaxes(title="Actual", autorange="reversed")


# In[ ]:





# In[ ]:


#show result


# In[ ]:


#add note (good bad...)


# In[ ]:


# show  Evaluate results


# In[ ]:


#add note (good bad...)


# In[ ]:





# In[ ]:


# summary...


# ### summary... 
# In this project, we tried to predict hotel reservation cancellations. We've Inspected the unique features, what influences the cancellation rate and what doesn't. In conclusion, we mananged to acheive a ~90% accuracy rate which is ~30% better than the naive models, and 5% higher than the most advanced model.  
# 

# In[ ]:





# In[ ]:




