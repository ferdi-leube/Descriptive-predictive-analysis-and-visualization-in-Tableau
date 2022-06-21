# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:54:59 2022

@author: leube
"""

import pandas as pd
import numpy as np
import datetime

data = pd.read_excel(r"C:\Users\leube\Ironhack\cloned_labs\module_2\Project_Week_6\Payments.xlsx")
payments = pd.DataFrame(data)
payments

payments['Date']
valuespdd = payments['Date'].unique()

# data cleaning experiments

trydte = '16-nov-21'
len(trydte)
value = list(trydte)
print(val)

dicdates = {'jan':1,'feb':2,'mar':3}

day = val[:2]
day1 = val[0] + day[1]
day1

year = val[-2:]
year1 = year[0] + year[1]
year1

month = val[3:6]
month
month1 = month[0] + month[1] + month[2]
month1


trydte = '16-mars-21'
val = list(trydte)


# results


list(valuespdd)
dicdates = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
for y in payments['Date'].tolist():
    if type(y) == str:
        val = [x.lower() for x in y]
        day = val[:2]
        day1 = val[0] + val[1]
    

        year = val[-2:]
        year1 = year[0] + year[1]
        
        month = val[3:6]
        month1 = month[0] + month[1] + month[2]
    
        correctdate = datetime.datetime(2000 + int(year1), dicdates[month1] , int(day1), 0, 0)
        print(y)
        print(correctdate)
        payments["Date"].replace({y: correctdate}, inplace=True)
                
                
        # payments['Date'] = np.where(payments['Date'] == y,correctdate,y)
    else:
        continue
    
payments['Date'].tolist()
payments['Date']


for x in payments['country code'].tolist():
    payments["country"].replace({x: countries[x]}, inplace=True)
    
    
# further formatting

# create unique column of ref + amount

payments
payments.columns

payments['str amount'] = payments["Initial Amount"].apply(str)

payments['unique ref + amount'] = payments['Reference'] + payments['str amount'] + payments['Creditor'] + payments['Currency']

del payments["'unique ref + amount'"]

# collect country code from bank account and add column with country names

payments['Bank'][0][4:6]
payments['country code'] = payments['Bank']

for x in range(len(payments['Bank'])):

    payments['country code'][x] = payments['Bank'][x][4:6]

payments['country code']



countries = {
	'AD': 'Andorra',
	'AE': 'United Arab Emirates',
	'AF': 'Afghanistan',
	'AG': 'Antigua & Barbuda',
	'AI': 'Anguilla',
	'AL': 'Albania',
	'AM': 'Armenia',
	'AN': 'Netherlands Antilles',
	'AO': 'Angola',
	'AQ': 'Antarctica',
	'AR': 'Argentina',
	'AS': 'American Samoa',
	'AT': 'Austria',
	'AU': 'Australia',
	'AW': 'Aruba',
	'AZ': 'Azerbaijan',
	'BA': 'Bosnia and Herzegovina',
	'BB': 'Barbados',
	'BD': 'Bangladesh',
	'BE': 'Belgium',
	'BF': 'Burkina Faso',
	'BG': 'Bulgaria',
	'BH': 'Bahrain',
	'BI': 'Burundi',
	'BJ': 'Benin',
	'BM': 'Bermuda',
	'BN': 'Brunei Darussalam',
	'BO': 'Bolivia',
	'BR': 'Brazil',
	'BS': 'Bahama',
	'BT': 'Bhutan',
	'BU': 'Burma (no longer exists)',
	'BV': 'Bouvet Island',
	'BW': 'Botswana',
	'BY': 'Belarus',
	'BZ': 'Belize',
	'CA': 'Canada',
	'CC': 'Cocos (Keeling) Islands',
	'CF': 'Central African Republic',
	'CG': 'Congo',
	'CH': 'Switzerland',
	'CI': 'Côte D\'ivoire (Ivory Coast)',
	'CK': 'Cook Iislands',
	'CL': 'Chile',
	'CM': 'Cameroon',
	'CN': 'China',
	'CO': 'Colombia',
	'CR': 'Costa Rica',
	'CS': 'Czechoslovakia (no longer exists)',
	'CU': 'Cuba',
	'CV': 'Cape Verde',
	'CX': 'Christmas Island',
	'CY': 'Cyprus',
	'CZ': 'Czech Republic',
	'DD': 'German Democratic Republic (no longer exists)',
	'DE': 'Germany',
	'DJ': 'Djibouti',
	'DK': 'Denmark',
	'DM': 'Dominica',
	'DO': 'Dominican Republic',
	'DZ': 'Algeria',
	'EC': 'Ecuador',
	'EE': 'Estonia',
	'EG': 'Egypt',
	'EH': 'Western Sahara',
	'ER': 'Eritrea',
	'ES': 'Spain',
	'ET': 'Ethiopia',
	'FI': 'Finland',
	'FJ': 'Fiji',
	'FK': 'Falkland Islands (Malvinas)',
	'FM': 'Micronesia',
	'FO': 'Faroe Islands',
	'FR': 'France',
	'FX': 'France, Metropolitan',
	'GA': 'Gabon',
	'GB': 'United Kingdom (Great Britain)',
	'GD': 'Grenada',
	'GE': 'Georgia',
	'GF': 'French Guiana',
	'GH': 'Ghana',
	'GI': 'Gibraltar',
	'GL': 'Greenland',
	'GM': 'Gambia',
	'GN': 'Guinea',
	'GP': 'Guadeloupe',
	'GQ': 'Equatorial Guinea',
	'GR': 'Greece',
	'GS': 'South Georgia and the South Sandwich Islands',
	'GT': 'Guatemala',
	'GU': 'Guam',
	'GW': 'Guinea-Bissau',
	'GY': 'Guyana',
	'HK': 'Hong Kong',
	'HM': 'Heard & McDonald Islands',
	'HN': 'Honduras',
	'HR': 'Croatia',
	'HT': 'Haiti',
	'HU': 'Hungary',
	'ID': 'Indonesia',
	'IE': 'Ireland',
	'IL': 'Israel',
	'IN': 'India',
	'IO': 'British Indian Ocean Territory',
	'IQ': 'Iraq',
	'IR': 'Islamic Republic of Iran',
	'IS': 'Iceland',
	'IT': 'Italy',
	'JM': 'Jamaica',
	'JO': 'Jordan',
	'JP': 'Japan',
	'KE': 'Kenya',
	'KG': 'Kyrgyzstan',
	'KH': 'Cambodia',
	'KI': 'Kiribati',
	'KM': 'Comoros',
	'KN': 'St. Kitts and Nevis',
	'KP': 'Korea, Democratic People\'s Republic of',
	'KR': 'Korea, Republic of',
	'KW': 'Kuwait',
	'KY': 'Cayman Islands',
	'KZ': 'Kazakhstan',
	'LA': 'Lao People\'s Democratic Republic',
	'LB': 'Lebanon',
	'LC': 'Saint Lucia',
	'LI': 'Liechtenstein',
	'LK': 'Sri Lanka',
	'LR': 'Liberia',
	'LS': 'Lesotho',
	'LT': 'Lithuania',
	'LU': 'Luxembourg',
	'LV': 'Latvia',
	'LY': 'Libyan Arab Jamahiriya',
	'MA': 'Morocco',
	'MC': 'Monaco',
	'MD': 'Moldova, Republic of',
	'MG': 'Madagascar',
	'MH': 'Marshall Islands',
	'ML': 'Mali',
	'MN': 'Mongolia',
	'MM': 'Myanmar',
	'MO': 'Macau',
	'MP': 'Northern Mariana Islands',
	'MQ': 'Martinique',
	'MR': 'Mauritania',
	'MS': 'Monserrat',
	'MT': 'Malta',
	'MU': 'Mauritius',
	'MV': 'Maldives',
	'MW': 'Malawi',
	'MX': 'Mexico',
	'MY': 'Malaysia',
	'MZ': 'Mozambique',
	'NA': 'Namibia',
	'NC': 'New Caledonia',
	'NE': 'Niger',
	'NF': 'Norfolk Island',
	'NG': 'Nigeria',
	'NI': 'Nicaragua',
	'NL': 'Netherlands',
	'NO': 'Norway',
	'NP': 'Nepal',
	'NR': 'Nauru',
	'NT': 'Neutral Zone (no longer exists)',
	'NU': 'Niue',
	'NZ': 'New Zealand',
	'OM': 'Oman',
	'PA': 'Panama',
	'PE': 'Peru',
	'PF': 'French Polynesia',
	'PG': 'Papua New Guinea',
	'PH': 'Philippines',
	'PK': 'Pakistan',
	'PL': 'Poland',
	'PM': 'St. Pierre & Miquelon',
	'PN': 'Pitcairn',
	'PR': 'Puerto Rico',
	'PT': 'Portugal',
	'PW': 'Palau',
	'PY': 'Paraguay',
	'QA': 'Qatar',
	'RE': 'Réunion',
	'RO': 'Romania',
	'RU': 'Russian Federation',
	'RW': 'Rwanda',
	'SA': 'Saudi Arabia',
	'SB': 'Solomon Islands',
	'SC': 'Seychelles',
	'SD': 'Sudan',
	'SE': 'Sweden',
	'SG': 'Singapore',
	'SH': 'St. Helena',
	'SI': 'Slovenia',
	'SJ': 'Svalbard & Jan Mayen Islands',
	'SK': 'Slovakia',
	'SL': 'Sierra Leone',
	'SM': 'San Marino',
	'SN': 'Senegal',
	'SO': 'Somalia',
	'SR': 'Suriname',
	'ST': 'Sao Tome & Principe',
	'SU': 'Union of Soviet Socialist Republics (no longer exists)',
	'SV': 'El Salvador',
	'SY': 'Syrian Arab Republic',
	'SZ': 'Swaziland',
	'TC': 'Turks & Caicos Islands',
	'TD': 'Chad',
	'TF': 'French Southern Territories',
	'TG': 'Togo',
	'TH': 'Thailand',
	'TJ': 'Tajikistan',
	'TK': 'Tokelau',
	'TM': 'Turkmenistan',
	'TN': 'Tunisia',
	'TO': 'Tonga',
	'TP': 'East Timor',
	'TR': 'Turkey',
	'TT': 'Trinidad & Tobago',
	'TV': 'Tuvalu',
	'TW': 'Taiwan, Province of China',
	'TZ': 'Tanzania, United Republic of',
	'UA': 'Ukraine',
	'UG': 'Uganda',
	'UM': 'United States Minor Outlying Islands',
	'US': 'United States of America',
	'UY': 'Uruguay',
	'UZ': 'Uzbekistan',
	'VA': 'Vatican City State (Holy See)',
	'VC': 'St. Vincent & the Grenadines',
	'VE': 'Venezuela',
	'VG': 'British Virgin Islands',
	'VI': 'United States Virgin Islands',
	'VN': 'Viet Nam',
	'VU': 'Vanuatu',
	'WF': 'Wallis & Futuna Islands',
	'WS': 'Samoa',
	'YD': 'Democratic Yemen (no longer exists)',
	'YE': 'Yemen',
	'YT': 'Mayotte',
	'YU': 'Yugoslavia',
	'ZA': 'South Africa',
	'ZM': 'Zambia',
	'ZR': 'Zaire',
	'ZW': 'Zimbabwe',
	'ZZ': 'Unknown or unspecified country'
}

for x in list(payments['country code'].unique()):
    print(countries[x])

payments['country'] = payments['country code']
payments['country']
for x in payments['country code'].tolist():
    payments["country"].replace({x: countries[x]}, inplace=True)
    
    
# upload csv

payments.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_2\preped_data.csv')

payments

# find average transaction time


pivot = payments.pivot_table(index=['unique ref + amount'], columns=['Status'], values=['Date'])
pivot

pivot.columns

calcpiv = pivot.transpose()
calcpiv

columns = list(calcpiv.columns)
maxes = [calcpiv[x].max() for x in columns]

minis = [calcpiv[x].min() for x in columns]

differences = []
for x in columns:
    if calcpiv.iloc[5][x] != np.nan:
        diff = maxes[columns.index[x]]-minis[columns.index[x]]
        differences.append(diff)
    else:
        diff = datetime.date(datetime.now())-minis[columns.index[x]]
        differences.append(diff)
differences


    
sum/len(list(payments['unique ref + amount'].unique()))

time_series = pd.Series(differences)
x = time_series.dt.days.sum()
x
avg = x/len(list(payments['unique ref + amount'].unique()))

avg*24

payments

# second approach


# add duration of transaction column


pivot = payments.pivot_table(index=['unique ref + amount'], columns=['Status'], values=['Date'])
pivot
pivot['duration'] = pivot.max(axis=1) - pivot.min(axis=1)

pivot['duration'] = pivot['duration'] + datetime.timedelta(days=1)

pivot['duration']



pivot['durration_format'] = pivot['duration'].dt.days

pivot

pivot['duration']
pivot.to_excel(r'C:\Users\leube\Ironhack\Ironprojects\Module_2\pivot.xlsx')





# status of payments forecast

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# df.groupby(df.date.dt.month)['sales'].max()


payments['Status'].unique()

filtered = payments[payments['Status'] == 'COMPLETED']
filtered.shape
filtered ['year'] = filtered['Date'].dt.year
filtered ['month'] = filtered['Date'].dt.month

ex = filtered.groupby(['year','month']).agg({'Status':'count'})
ex.reset_index(inplace=True)


X = sm.add_constant(ex[['year','month']])
Y = ex['Status']
model = sm.OLS(Y,X)
results = model.fit()
predictions = results.predict(X)

residuals = [Y[i] - predictions[i] for i in range(len(Y))]

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(Y, predictions))
print(r2_score(Y, predictions))
print(np.sqrt(mean_squared_error(Y, predictions)))

# vary low r squarred value for how much of the data is represented by the model,
# therefor predicting the number of complet transactions is not likely to be a very
# realistic model, reason for this is probably the very low amount of data available


print_model = results.summary()
print(print_model)

# initial amount forecast

filt = payments.groupby('Date').agg({'Initial Amount':'mean'})
filt.reset_index(inplace=True)
filt ['year'] = filt['Date'].dt.year
filt ['month'] = filt['Date'].dt.month

char = filt.groupby(['year','month']).agg({'Initial Amount':'mean'})
char.reset_index(inplace=True)
char

# regression model

X = sm.add_constant(char[['year','month']])
Y = char['Initial Amount']
model = sm.OLS(Y,X)
results = model.fit()
predictions = results.predict(X)

residuals = [Y[i] - predictions[i] for i in range(len(Y))]

print(mean_squared_error(Y, predictions))
print(r2_score(Y, predictions))
print(np.sqrt(mean_squared_error(Y, predictions)))


# same problem with regression model due to limited amount of data but lets try to 
# forecast anyway for demonstration purposes

print_model = results.summary()
print(print_model)

def predict_amount(year, month):
     return -8.358e+07+4.135e+04*year+1.021e+04*month
    
# try
predict_amount(2022,2)


year = 2022
months = [2,3,4]

for month in months:
    char = char.append({'year': year,'month':month,'Initial Amount':predict_amount(year,month)}, ignore_index=True)
    
char

char.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_2\initalam_forecast2.csv', index=False, sep=';')
#checks

payments ['month'] = payments['Date'].dt.month
payments['month'].unique()


# troubles forecast

filteredtr = pivot[pivot['durration_format'] >= 2]
filteredtr

# total charges calculations 


charge = payments[['Initial Amount', 'Charges','unique ref + amount']]
charge
calcs = charge.groupby(['unique ref + amount','Initial Amount']).agg({'Charges':'prod'})

calcs.shape
calcs.reset_index(inplace=True)
calcs.shape
calcs['end charges'] = calcs['Initial Amount'] * calcs['Charges']

calcs.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_2\charges.csv')



# check country creditor

payments.pivot_table(index=['Creditor'], columns='country', aggfunc='count')



# get current state of each individual transaction

payments['Status'].unique()



conditions3 = [payments['Status']=='NEW',
               payments['Status']=='PENDING',
               payments['Status']=='PROCESSING',
               payments['Status']=='DELIVERED',
               payments['Status']=='CANCELLED',
               payments['Status']=='COMPLETED'
]
choices3 = [1,
            2,
            3,
            4,
            5,
            6]
payments['Status_encoded'] = np.select(conditions3, choices3, 'huge')

payments.columns
payments['Status_encoded']

transactions = list(payments['unique ref + amount'].unique())
currentstate = []
for x in transactions:
    filteredtr = payments[payments['unique ref + amount']==x]
    current_status = filteredtr['Status_encoded'].max()
    currentstate.append(current_status)
    
current_state_transactions = pd.DataFrame({'Transaction':transactions,'current_status':currentstate})
current_state_transactions    

conditions3 = [current_state_transactions['current_status']=='1',
              current_state_transactions['current_status']=='2',
              current_state_transactions['current_status']=='3',
              current_state_transactions['current_status']=='4',
              current_state_transactions['current_status']=='5',
              current_state_transactions['current_status']=='6'
              ]

current_state_transactions['current_status'].unique()
choices3 = ['NEW',
            'PENDING',
            'PROCESSING',
            'DELIVERED',
            'CANCELLED',
            'COMPLETED']
current_state_transactions['current_status_inverse'] = np.select(conditions3, choices3, 'huge')
    
current_state_transactions

current_state_transactions.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_2\Descriptive-predictive-analysis-and-visualization-in-Tableau\status_current.csv') 
    
    
    
    
    
