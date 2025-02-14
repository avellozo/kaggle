
import sys
sys.path.append('/home/augusto/projects/kaggle')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer 
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import BaseEstimator, TransformerMixin
from lifelines.utils import concordance_index

from preprocess import ConvertObjectToCategoricalTransformer, DataFrameMinMaxScaler, DropColumnsTransformer, DropHighCorrelationColumnsTransformer, DropSingleValueColumnsTransformer, FixBoolTypeTransformer, OneHotEncodeCategoricalTransformer, RemoveOutliersPercentilesTransformer, ReplaceTransformer, TreatMissingValuesTransformer
from tools import calc_score, createSubmission

class CoxPHWrapper(BaseEstimator, RegressorMixin):
    """
    Um wrapper do CoxPHFitter para que ele se comporte como um estimador do scikit-learn.
    Espera que a variável alvo (y) seja um DataFrame contendo duas colunas:
      - Uma com o tempo (por exemplo, 'efs_time')
      - Outra com o status do evento (por exemplo, 'efs', onde 1 = evento, 0 = censurado)
    """
    def __init__(self, duration_col='efs_time', event_col='efs', **cox_kwargs):
        self.duration_col = duration_col
        self.event_col = event_col
        self.cox_kwargs = cox_kwargs
        self.cox_model = None

    def fit(self, X, y):
        """
        X deve ser um DataFrame (ou pode ser convertido para DataFrame) com as covariáveis;
        y deve ser um DataFrame com as colunas [duration_col, event_col].
        """
        # Converte X para DataFrame, caso não seja
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, index=y.index)
        
        # Verifica se y é um DataFrame e possui as colunas necessárias
        if not isinstance(y, pd.DataFrame) or \
           (self.duration_col not in y.columns or self.event_col not in y.columns):
            raise ValueError(f"y deve ser um DataFrame contendo as colunas '{self.duration_col}' e '{self.event_col}'")
        if X.isnull().any().any():
            print("Há NaNs no X:")
            print(df.isnull().sum())
            raise ValueError("Encontrados NaNs no X.")
        if y.isnull().any().any():
            print("Há NaNs no y:")
            print(df.isnull().sum())
            raise ValueError("Encontrados NaNs no y.")             
        # Junta X e y em um único DataFrame para o lifelines
        df = pd.concat([X, y], axis=1)
        if df.isnull().any().any():
            print("Há NaNs no DataFrame combinado:")
            print(df.isnull().sum())
            raise ValueError("Encontrados NaNs após a concatenação de X e y.")
        self.cox_model = CoxPHFitter(**self.cox_kwargs)
        self.cox_model.fit(df, duration_col=self.duration_col, event_col=self.event_col)
        return self

    def predict(self, X):
        """
        Para predições, podemos retornar o risco parcial (partial hazard) ou outra métrica de interesse.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Retorna o hazard parcial para cada observação.
        return self.cox_model.predict_partial_hazard(X)

"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

# Carregar os dados
data = pd.read_csv('/home/augusto/projects/kaggle/cibmtr/input/equity-post-HCT-survival-predictions/train.csv') 

# Separar variáveis preditoras (features) e variável alvo (target)
X = data.drop(['efs', 'efs_time'], axis=1)  # Features
y = data[['efs', 'efs_time']]  # Target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Criar um df com os dados de teste para obter o score, deve conter as linhas de data com o mesmo índice de X_test 
solution_test = data.loc[X_test.index]

cols_to_drop = ['hla_match_c_high', 'hla_match_c_low', 'hla_match_dqb1_low', 'dri_score_High', 
 'dri_score_High - TED AML case <missing cytogenetics', 'dri_score_Intermediate - TED AML case <missing cytogenetics', 
 'dri_score_Low', 'dri_score_Missing disease status', 'dri_score_N/A - disease not classifiable', 'dri_score_N/A - non-malignant indication', 
 'dri_score_N/A - pediatric', 'dri_score_TBD cytogenetics', 'dri_score_Very high', 'dri_score_nan', 
 'cyto_score_Favorable', 'cyto_score_Intermediate', 'cyto_score_Normal', 'cyto_score_Not tested', 
 'cyto_score_Other', 'cyto_score_TBD', 'tbi_status_TBI + Cy +- Other', 'tbi_status_TBI +- Other, -cGy, fractionated', 
 'tbi_status_TBI +- Other, -cGy, single', 'tbi_status_TBI +- Other, -cGy, unknown dose', 
 'tbi_status_TBI +- Other, <=cGy', 'tbi_status_TBI +- Other, >cGy', 'tbi_status_TBI +- Other, unknown dose', 
 'prim_disease_hct_AI', 'prim_disease_hct_AML', 'prim_disease_hct_CML', 'prim_disease_hct_HD', 'prim_disease_hct_HIS', 
 'prim_disease_hct_IEA', 'prim_disease_hct_IIS', 'prim_disease_hct_IMD', 'prim_disease_hct_IPA', 'prim_disease_hct_MDS', 
 'prim_disease_hct_MPN', 'prim_disease_hct_NHL', 'prim_disease_hct_Other acute leukemia', 
 'prim_disease_hct_Other leukemia', 'prim_disease_hct_PCD', 'prim_disease_hct_SAA', 'prim_disease_hct_Solid tumor', 
 'cmv_status_+/-', 'cmv_status_-/+', 'cmv_status_-/-', 'cmv_status_nan', 'tce_imm_match_G/B', 'tce_imm_match_G/G', 
 'tce_imm_match_H/B', 'tce_imm_match_H/H', 'tce_imm_match_P/B', 'tce_imm_match_P/G', 'tce_imm_match_P/H', 
 'cyto_score_detail_Favorable', 'cyto_score_detail_Not tested', 'cyto_score_detail_Poor', 'cyto_score_detail_TBD', 
 'conditioning_intensity_N/A, F(pre-TED) not submitted', 'conditioning_intensity_NMA', 
 'conditioning_intensity_No drugs reported', 'conditioning_intensity_TBD', 'conditioning_intensity_nan', 
 'ethnicity_Hispanic or Latino', 'ethnicity_Non-resident of the U.S.', 'ethnicity_Not Hispanic or Latino', 
 'ethnicity_nan', 'tce_match_Fully matched', 'tce_match_GvH non-permissive', 'tce_match_HvG non-permissive', 
 'tce_match_Permissive', 'gvhd_proph_CDselect +- other', 'gvhd_proph_CDselect alone', 
 'gvhd_proph_CSA + MMF +- others(not FK)', 'gvhd_proph_CSA + MTX +- others(not MMF,FK)', 
 'gvhd_proph_CSA +- others(not FK,MMF,MTX)', 'gvhd_proph_CSA alone', 'gvhd_proph_Cyclophosphamide +- others', 
 'gvhd_proph_Cyclophosphamide alone', 'gvhd_proph_FK+ MTX +- others(not MMF)', 'gvhd_proph_FK+- others(not MMF,MTX)', 
 'gvhd_proph_FKalone', 'gvhd_proph_No GvHD Prophylaxis', 'gvhd_proph_Other GVHD Prophylaxis', 
 'gvhd_proph_Parent Q = yes, but no agent', 'gvhd_proph_TDEPLETION +- other', 'gvhd_proph_TDEPLETION alone', 
 'gvhd_proph_nan', 'sex_match_F-F', 'sex_match_M-F', 'sex_match_nan', 'race_group_American Indian or Alaska Native', 
 'race_group_Asian', 'race_group_Black or African-American', 'race_group_More than one race', 
 'race_group_Native Hawaiian or other Pacific Islander', 'race_group_White', 
 'tce_div_match_Bi-directional non-permissive', 'tce_div_match_GvH non-permissive', 
 'tce_div_match_HvG non-permissive', 'donor_related_Multiple donor (non-UCB)', 'donor_related_nan'] 


# Criar um pipeline para o pré-processamento dos dados
# -------------------------------
# CRIAÇÃO DO PIPELINE
# -------------------------------

# Note que a ordem segue as etapas:
# 1. Remover colunas desnecessárias ('ID')
# 2. Remover colunas com apenas um valor
# 3. Substituir 'Not done' por NaN
# 4. Corrigir tipo booleano
# 5. Converter colunas object para categorical
# 6. One-Hot Encoding para as colunas categóricas
# 7. Remover colunas com alta correlação
# 8. Remover outliers (utilizando percentis)
# 9. Tratar os missing values
# 10. Aplicar MinMaxScaler

preprocessing_pipeline = Pipeline(steps=[
    ('drop_columns_ID', DropColumnsTransformer(columns_to_drop=['ID'])),
    ('drop_single_value', DropSingleValueColumnsTransformer()),
    ('replace_not_done', ReplaceTransformer(value_to_replace='Not done', replacement=np.nan)),
    ('fix_bool', FixBoolTypeTransformer()),
    ('convert_to_categorical', ConvertObjectToCategoricalTransformer(columns_to_exclude=['race_group'])),
    ('ohe', OneHotEncodeCategoricalTransformer()),
    ('drop_high_corr', DropHighCorrelationColumnsTransformer(threshold=0.80, undrop_cols=['race_group'])),
    ('remove_outliers', RemoveOutliersPercentilesTransformer(percentile=0.10)),
    ('treat_missing', TreatMissingValuesTransformer()),
    ('scaler', DataFrameMinMaxScaler()),
    ('drop_columns_converge', DropColumnsTransformer(columns_to_drop=cols_to_drop)),
])


preprocessing_pipeline.fit(X_train, y_train)
X_train_preprocessed = preprocessing_pipeline.transform(X_train)
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Treinar o modelo
cox_model = CoxPHFitter(l1_ratio=0.1, penalizer=0.1, strata=['race_group'])
df = pd.concat([X_train_preprocessed, y_train], axis=1)
cox_model.fit(df, duration_col='efs_time', event_col='efs', show_progress=True)
y_pred = cox_model.predict_partial_hazard(X_test_preprocessed)

# Fazer previsões

# y_pred = pipeline.predict(X_test)

X_test['ID'] = solution_test['ID']
# y_pred['ID'] = solution_test['ID']
submission = createSubmission(X_test, y_pred)
score = calc_score(solution_test, submission)
print(f"Score: {score}")

