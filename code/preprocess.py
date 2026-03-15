from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import MinMaxScaler


KEY_COLS = ['id_student', 'code_module', 'code_presentation']


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'OULAD'


def _target_col(df: pd.DataFrame) -> str:
    for col in ['final-result', 'final_result']:
        if col in df.columns:
            return col
    raise ValueError('Target column not found. Expected one of: final-result, final_result')


def pp_studentInfo() -> pd.DataFrame:
    data_dir = _data_dir()
    sample = pd.read_csv(data_dir / 'studentInfo.csv')

    onehot_cols = ['gender', 'region', 'highest_education', 'disability']
    onehot_cols = [col for col in onehot_cols if col in sample.columns]

    age_map = {'0-35': 0, '35-55': 1, '55<=': 2}
    imd_order = ['0-10%', '10-20', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    imd_map = {value: idx for idx, value in enumerate(imd_order)}

    if 'age_band' in sample.columns:
        sample['age_band'] = sample['age_band'].map(age_map).fillna(-1).astype('int8')
    if 'imd_band' in sample.columns:
        sample['imd_band'] = sample['imd_band'].map(imd_map).fillna(-1).astype('int8')

    encoded = pd.get_dummies(sample, columns=onehot_cols, dummy_na=False)
    encoded.to_csv(data_dir / 'sI_pp.csv', index=False)
    return encoded


def aggr_pp_assessment() -> pd.DataFrame:
    data_dir = _data_dir()

    assessment = pd.read_csv(data_dir / 'assessments.csv')
    student_assessment = pd.read_csv(data_dir / 'studentAssessment.csv')

    merge_df = pd.merge(student_assessment, assessment, on='id_assessment', how='left')

    merge_df['has_due_date'] = merge_df['date'].notna().astype('int8')
    merge_df['score_missing'] = merge_df['score'].isna().astype('int8')
    merge_df['date'] = merge_df['date'].fillna(merge_df['date'].median())
    merge_df['score'] = merge_df['score'].fillna(merge_df['score'].median())

    merge_df['submission_delay'] = merge_df['date_submitted'] - merge_df['date']
    merge_df['is_late'] = (merge_df['submission_delay'] > 0).astype('int8')
    merge_df['is_early'] = (merge_df['submission_delay'] < 0).astype('int8')

    delay_low = merge_df['submission_delay'].quantile(0.01)
    delay_high = merge_df['submission_delay'].quantile(0.99)
    merge_df['submission_delay_capped'] = merge_df['submission_delay'].clip(lower=delay_low, upper=delay_high)

    merge_df['weighted_score'] = merge_df['score'] * (merge_df['weight'] * 0.01)
    merge_df['weighted_score_norm'] = MinMaxScaler().fit_transform(merge_df[['weighted_score']])

    student_features = merge_df.groupby(KEY_COLS, as_index=False).agg(
        assessment_count=('id_assessment', 'count'),
        score_mean=('score', 'mean'),
        score_median=('score', 'median'),
        score_std=('score', 'std'),
        weighted_score_sum=('weighted_score', 'sum'),
        weighted_score_norm_mean=('weighted_score_norm', 'mean'),
        late_ratio=('is_late', 'mean'),
        early_ratio=('is_early', 'mean'),
        delay_mean=('submission_delay_capped', 'mean'),
        missing_due_ratio=('has_due_date', lambda s: 1 - s.mean()),
        missing_score_ratio=('score_missing', 'mean'),
    )

    type_counts = (
        merge_df.pivot_table(
            index=KEY_COLS,
            columns='assessment_type',
            values='id_assessment',
            aggfunc='count',
            fill_value=0,
        )
        .add_prefix('assessment_count_')
        .reset_index()
    )

    student_features = pd.merge(student_features, type_counts, on=KEY_COLS, how='left').fillna(0)
    student_features.to_csv(data_dir / 'sA_merged_pp.csv', index=False)
    return student_features


def aggr_pp_Vle(chunksize: int = 500_000) -> pd.DataFrame:
    data_dir = _data_dir()

    # latest assessment date per module/presentation used as the click cutoff
    assessments = pd.read_csv(
        data_dir / 'assessments.csv',
        usecols=['code_module', 'code_presentation', 'date'],
    )
    cutoff = (
        assessments.dropna(subset=['date'])
        .groupby(['code_module', 'code_presentation'], as_index=False)['date']
        .max()
        .rename(columns={'date': 'cutoff_date'})
    )

    # activity_type lookup from the small vle table
    vle = pd.read_csv(
        data_dir / 'vle.csv',
        usecols=['id_site', 'activity_type'],
        dtype={'id_site': 'int32'},
    ).drop_duplicates(subset=['id_site'])

    partial_frames = []
    reader = pd.read_csv(
        data_dir / 'studentVle.csv',
        chunksize=chunksize,
        usecols=['code_module', 'code_presentation', 'id_site', 'id_student', 'date', 'sum_click'],
        dtype={
            'id_site': 'int32',
            'id_student': 'int32',
            'date': 'int16',
            'sum_click': 'float32',
        },
    )

    for chunk in reader:
        # drop clicks that happen after the last assessment date
        chunk = chunk.merge(cutoff, on=['code_module', 'code_presentation'], how='left')
        chunk = chunk[chunk['date'] <= chunk['cutoff_date']].drop(columns=['cutoff_date'])

        # attach activity type
        chunk = chunk.merge(vle, on='id_site', how='left')
        chunk['activity_type'] = chunk['activity_type'].fillna('unknown')

        # sum clicks per student per activity type
        pivot = (
            chunk.pivot_table(
                index=KEY_COLS,
                columns='activity_type',
                values='sum_click',
                aggfunc='sum',
                fill_value=0,
            )
            .add_prefix('clicks_')
            .reset_index()
        )
        partial_frames.append(pivot)

    combined = pd.concat(partial_frames, ignore_index=True)

    click_cols = [col for col in combined.columns if col.startswith('clicks_')]
    behavior_features = combined.groupby(KEY_COLS, as_index=False)[click_cols].sum()

    behavior_features.to_csv(data_dir / 'sV_merged_pp.csv', index=False)
    return behavior_features


def build_train_test_sets(test_size: float = 0.2, random_state: int = 69):
    data_dir = _data_dir()

    student_info = pd.read_csv(data_dir / 'sI_pp.csv')
    performance = pd.read_csv(data_dir / 'sA_merged_pp.csv')
    behavior = pd.read_csv(data_dir / 'sV_merged_pp.csv')

    sample = pd.merge(student_info, performance, on=KEY_COLS, how='left')
    sample = pd.merge(sample, behavior, on=KEY_COLS, how='left')

    target = _target_col(sample)
    y = sample[target]
    X = sample.drop(columns=[target])

    numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    X = pd.get_dummies(X, columns=non_numeric_cols, dummy_na=False)

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    counts = y_train.value_counts(normalize=True)
    minority_class = counts.min()
    if minority_class < 0.2 and y_train.nunique() > 1:
        smote = SMOTE(random_state=67)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    X_train.to_csv(data_dir / 'X_train.csv', index=False)
    y_train.to_csv(data_dir / 'y_train.csv', index=False)
    X_test.to_csv(data_dir / 'X_test.csv', index=False)
    y_test.to_csv(data_dir / 'y_test.csv', index=False)



def main():
    pp_studentInfo()
    aggr_pp_assessment()
    aggr_pp_Vle()
    return build_train_test_sets()


if __name__ == '__main__':
    main()
