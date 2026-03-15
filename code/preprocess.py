import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd 
import imblearn.over_sampling as imblearn
from imblearn.over_sampling import SMOTE

def split_studentInfo(sample, number):
    # split into train and test sets
    X = sample.drop(columns=['final-result'])
    y = sample['final-result']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=69,
        stratify=y,
    )

    # check for imbalance and apply SMOTE to training set only
    counts = y_train.value_counts(normalize=True)
    minority_class = counts.min()
    if minority_class < 0.2:
        print('imbalanced sample; fixing')
        smote = SMOTE(random_state=67)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        print('balanced sample; no need to fix')
        X_train_res, y_train_res = X_train, y_train

    return X_train_res, X_test, y_train_res, y_test

def pp_studentInfo(sample, num):
    cols_to_1hot_encode = ['gender', 'region', 'highest-education', 'disability']
    cols_to_ordinal_encode = ['age_band', 'imd_band']   
    ordinal_rank = [['0-35', '35-55', '55<='], ['0-10%', '10-20', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']]
    preprocessor = preprocessing.ColumnTransformer(
        rtansformers=[
            ('onehot', preprocessing.OneHotEncoder(), cols_to_1hot_encode),
            ('ordinal', preprocessing.OrdinalEncoder(categories=ordinal_rank), cols_to_ordinal_encode)
        ],
        remainder='passthrough'
    )
    sample = preprocessor.fit_resample(sample)
    
    return split_studentInfo(sample, num)