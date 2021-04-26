# # Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Load Data

df = pd.read_csv('data_fintech.csv')


# # Review Data

df.info()
df.shape
describe = df.describe()

# # Data Cleaning
# ## Checking Duplicates Data

df.drop_duplicates(inplace = True)

# ## Checking Data Types


#removes space in hour
df['hour'] = df['hour'].str.strip()
df['first_open'] = pd.to_datetime(df['first_open'])
df['enrolled_date'] = df['enrolled_date'].astype('datetime64')
df['screen_list'] = df.screen_list.astype(str) + ','
df.drop(columns= ['numscreens'], inplace=True)
df['numscreens'] = df.screen_list.str.count(',')
df.dtypes


# ## Checking Missing Values

df.isnull().sum()

# ### Checking Missing Value Distribution Plot
#sns.distplot(df[''])
#df.dtypes

# ## Impute Missing Values

#Separating categorical and numerical columns


#for col in df:
#    if df[col].isnull().any():
#        if col in cat_cols:
#            df[col] = df[col].fillna(df[col].mode()[0])
#        else:
#            df[col] = df[col].fillna(df[col].median())


#df.isnull().sum()

# ## Outlier Detection


# ### 


# ## Handle Inconsistent Data



# # Feature Engineering
# ## Binning Variable

#Binning variables become categorical variables



dt = pd.read_csv('top_screens.csv')
top_screens = dt['top_screens']

for i in top_screens:
    df[i] = df.screen_list.str.contains(i).astype('int64')
    df['screen_list'] = df.screen_list.str.replace(i + ',', '')

#Merge the same screen (Funelling)
screen_loan = ['Loan',
               'Loan2',
               'Loan3',
               'Loan4']
df['num_loan'] = df[screen_loan].sum(axis=1) #axis 1 for count all loans in rows
df.drop(columns = screen_loan, inplace = True)

screen_saving = ['Saving1',
               'Saving2',
               'Saving2Amount',
               'Saving4',
               'Saving5',
               'Saving6',
               'Saving7',
               'Saving8',
               'Saving9',
               'Saving10']
df['num_saving'] = df[screen_saving].sum(axis=1) #axis 1 for count all loans in rows
df.drop(columns = screen_saving, inplace = True)

screen_credit = ['Credit1',
               'Credit2',
               'Credit3',
               'Credit3Container',
               'Credit3Dashboard']
df['num_credit'] = df[screen_credit].sum(axis=1) #axis 1 for count all loans in rows
df.drop(columns = screen_credit, inplace = True)

screen_cc = ['CC1',
             'CC1Category',
             'CC3']
df['num_cc'] = df[screen_cc].sum(axis=1) #axis 1 for count all loans in rows
df.drop(columns = screen_cc, inplace = True)

# ## Encoding Variables
from sklearn.preprocessing import LabelEncoder

#Separating categorical and numerical columns was done while imputing missing values
Id_col     = ['user']
target_col = ['enrolled']
dt_cols    = ['first_open','hour','enrolled_date']
str_col   = ['screen_list']
cat_cols   = df.nunique()[df.nunique() <= 10].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col + dt_cols]
num_cols   = [x for x in df.columns if x not in cat_cols + target_col 
              + Id_col + dt_cols + str_col]
#Binary columns with 2 values
bin_cols   = df.nunique()[df.nunique() == 2].keys().tolist()

#multinomial columns
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    df[i] = le.fit_transform(df[i])
    
#Label encoding for nominal multi value columns
df = pd.get_dummies(data = df,columns = multi_cols,drop_first=True)

# # Data Partition

from sklearn.model_selection import train_test_split

##partition data into data training and data testing
train,test = train_test_split(df,test_size = .20 ,random_state = 111)
    
##seperating dependent and independent variables on training and testing data
cols    = [i for i in df.columns if i not in Id_col + target_col 
           + dt_cols + str_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)
# # SMOTE

#target column value count
train_Y['enrolled'].value_counts()
#imbalance data

from imblearn.over_sampling import SMOTE

#handle imbalance class using oversampling minority class with smote method
os = SMOTE(sampling_strategy='minority',random_state = 123,k_neighbors=5)
train_smote_X,train_smote_Y = os.fit_resample(train_X,train_Y)
train_smote_X = pd.DataFrame(data = train_smote_X,columns=cols)
train_smote_Y = pd.DataFrame(data = train_smote_Y,columns=target_col)

#Proportion after smote
train_smote_Y['enrolled'].value_counts()


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = 'liblinear',
                                penalty = 'l1')
classifier.fit(train_X, train_Y)

y_pred = classifier.predict(test_X)

#Evaluate model using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(test_Y, y_pred)
print(classification_report(test_Y, y_pred))

#Using accuracy_score
evaluasi = accuracy_score(test_Y, y_pred)
print('Accuracy:{:.2f}'.format(evaluasi*100))

#Seaborn for CM
cm_label = pd.DataFrame(cm, columns = np.unique(test_Y),
                        index = np.unique(test_Y))
cm_label.index.name = 'Aktual'
cm_label.columns.name = 'Prediksi'
sns.heatmap(cm_label, annot=True, cmap = 'Reds', fmt = 'g')

#Validate data with 10-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, 
                             X = train_smote_X, y = train_smote_Y,
                             cv = 10)
accuracies.mean()
accuracies.std()
accuracies.std()
print('Accuracy Logistic Regression = {:.2f} +/- {:.2f}'.format(accuracies.mean()*100,
                                                               accuracies.std()*100))