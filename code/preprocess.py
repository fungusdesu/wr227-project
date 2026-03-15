import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import imblearn.over_sampling as imblearn
from imblearn.over_sampling import SMOTE
from pathlib import Path

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
    #1hot and ordinal some cols to numeric values
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

    return sample

#aggregate studentAssessment and assessment tables with key id_assessment, then pp the merged table, then save the merged table as csv for later use
def aggr_pp_assessment():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / 'OULAD'

    assessment = pd.read_csv(data_dir / 'assessments.csv')
    student_assessment = pd.read_csv(data_dir / 'studentAssessment.csv')

    merge_df = pd.merge(student_assessment, assessment, on='id_assessment', how='left')

    # missing-value handling
    merge_df['has_due_date'] = merge_df['date'].notna().astype(int)
    merge_df['score_missing'] = merge_df['score'].isna().astype(int)
    merge_df['date'] = merge_df['date'].fillna(merge_df['date'].median())
    merge_df['score'] = merge_df['score'].fillna(merge_df['score'].median())

    # time-based features
    merge_df['submission_delay'] = merge_df['date_submitted'] - merge_df['date']
    merge_df['is_late'] = (merge_df['submission_delay'] > 0).astype(int)
    merge_df['is_early'] = (merge_df['submission_delay'] < 0).astype(int)

    # cap extreme delays to reduce outlier impact
    low_q = merge_df['submission_delay'].quantile(0.01)
    high_q = merge_df['submission_delay'].quantile(0.99)
    merge_df['submission_delay_capped'] = merge_df['submission_delay'].clip(lower=low_q, upper=high_q)

    # weighted score + normalization
    merge_df['weighted_score'] = merge_df['score'] * (merge_df['weight'] * 0.01)
    merge_df['weighted_score_norm'] = MinMaxScaler().fit_transform(merge_df[['weighted_score']])

    merge_df.to_csv(data_dir / 'sA_merged_pp.csv', index=False)
  




def aggr_pp_Vle():

    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / 'OULAD'

    student_vle = pd.read_csv(data_dir / 'studentVle.csv')
    vle = pd.read_csv(data_dir / 'vle.csv')

    merge_df = pd.merge(student_vle, vle, on='id_site', how='left')

    behavior_features = merge_df.pivot_table(
    index=['id_student', 'code_module', 'code_presentation'],
    columns='activity_type',
    values='sum_click',
    aggfunc='sum').fillna(0)

    behavior_scaled = pd.DataFrame(MinMaxScaler().fit_transform(behavior_features), columns=behavior_features.columns, index=behavior_features.index).reset_index()

    behavior_scaled.to_csv(data_dir / 'sV_behavior_scaled.csv', index=False)



