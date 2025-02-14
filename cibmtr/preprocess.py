import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from cibmtr.tools import basic_treat_missing_numbers_booleans, col_is_bool, convert_object_columns_to_categorical, convert_to_bool, fix_bol_type, ohe_catCols

# -------------------------------
# TRANSFORMADORES CUSTOMIZADOS
# -------------------------------

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove colunas desnecessárias."""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Garante que X seja DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class DropSingleValueColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove colunas que possuem apenas um valor único."""
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.cols_to_drop_ = [col for col in X.columns if X[col].nunique() <= 1]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_, errors='ignore')


class ReplaceTransformer(BaseEstimator, TransformerMixin):
    """Substitui o texto 'Not done' por NaN."""
    def __init__(self, value_to_replace='Not done', replacement=np.nan):
        self.value_to_replace = value_to_replace
        self.replacement = replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.replace(self.value_to_replace, self.replacement)


class FixBoolTypeTransformer(BaseEstimator, TransformerMixin):
    # Converte colunas para boolean quando possível.
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.cols_to_boolean_ = [col for col in X.columns if col_is_bool(X, col)]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()
        for col in self.cols_to_boolean_ :
            df[col] = df[col].apply(convert_to_bool).astype('boolean')
        return df 


class ConvertObjectToCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Converte colunas do tipo object para categorical, exceto as colunas informadas."""
    def __init__(self, columns_to_exclude=[]):
        self.columns_to_exclude = columns_to_exclude

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()        
        # Seleciona as colunas do DataFrame que são do tipo 'object'
        object_columns = df.select_dtypes(include='object')

        # Identifica as colunas que devem ser convertidas para 'category', excluindo aquelas
        # presentes na lista columns_to_exclude.
        columns_to_convert = [
            column for column in object_columns.columns if column not in self.columns_to_exclude
        ]
        self.columns_to_convert_ = columns_to_convert
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()
        # Itera sobre a lista de colunas a serem convertidas e realiza a conversão para 'category'.
        for column in self.columns_to_convert_:
            df[column] = df[column].astype('category')
        return df

class OneHotEncodeCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer que aplica One-Hot Encoding para colunas categóricas com mais de 2 categorias
    e Label Encoding para colunas categóricas com até 2 categorias. Durante o fit, ele aprende quais
    colunas transformar e armazena os respectivos encoders para replicar a transformação no transform.
    """
    def __init__(self):
        # Dicionários para armazenar os encoders e informações das colunas processadas
        self.encoders_ = {}              # encoder para cada coluna
        self.onehot_columns_ = []        # lista de colunas que serão OHE
        self.label_columns_ = []         # lista de colunas que serão label encoded
        self.onehot_feature_names_ = {}  # para cada coluna OHE, armazena os nomes das novas colunas geradas

    def fit(self, X, y=None):
        # Garante que X seja um DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Seleciona as colunas categóricas (assumindo que já estejam com o dtype 'category')
        categorical_cols = X.select_dtypes(include='category').columns.tolist()
        
        self.encoders_ = {}
        self.onehot_columns_ = []
        self.label_columns_ = []
        self.onehot_feature_names_ = {}
        
        for col in categorical_cols:
            num_categories = X[col].nunique()
            if num_categories > 2:
                # Configura o OneHotEncoder para essa coluna
                encoder = OneHotEncoder(drop="if_binary", handle_unknown='infrequent_if_exist', sparse_output=False)
                encoder.fit(X[[col]])
                self.encoders_[col] = encoder
                self.onehot_columns_.append(col)
                # Salva os nomes das colunas geradas: ex: "col_categoria1", "col_categoria2", etc.
                self.onehot_feature_names_[col] = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            else: # <= 2
                # Configura o LabelEncoder para essa coluna
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders_[col] = encoder
                self.label_columns_.append(col)
        return self

    def transform(self, X):
        # Garante que X seja um DataFrame e cria uma cópia para evitar alterações no original
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        
        # Aplica One-Hot Encoding nas colunas registradas
        for col in self.onehot_columns_:
            if col in X.columns:
                encoder = self.encoders_[col]
                encoded_array = encoder.transform(X[[col]])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.onehot_feature_names_[col],
                    index=X.index
                )
                # Remove a coluna original e concatena as novas colunas codificadas
                X = X.drop(columns=[col])
                X = pd.concat([X, encoded_df], axis=1)
        
        # Aplica Label Encoding nas colunas registradas
        for col in self.label_columns_:
            if col in X.columns:
                encoder = self.encoders_[col]
                # O LabelEncoder espera uma série 1D
                X[col] = encoder.transform(X[col])
        return X

class DropHighCorrelationColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove colunas com alta correlação."""
    def __init__(self, threshold=0.95, undrop_cols=[]):
        self.threshold = threshold
        self.undrop_cols = undrop_cols

    def fit(self, X, y=None):
        # Garante que X seja um DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X
        
        # Calcular a matriz de correlação absoluta
        correlation_matrix = df.drop(columns=self.undrop_cols).corr().abs()

        # Obter apenas o triângulo superior da matriz de correlação para evitar duplicatas
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Lista para armazenar colunas a serem removidas
        to_drop = []

        # Iterar pelas colunas no triângulo superior
        for column in upper_triangle.columns:
            # Encontrar colunas altamente correlacionadas com a coluna atual
            highly_corr_cols = upper_triangle.index[upper_triangle[column] > self.threshold].tolist()

            # Iterar pelas colunas correlacionadas
            for corr_col in highly_corr_cols:
                # Contar valores ausentes em ambas as colunas
                column_na_qtty = df[column].isna().sum()
                corr_col_na_qtty = df[corr_col].isna().sum()

                # Decidir qual coluna remover com base na quantidade de valores ausentes
                if (column_na_qtty > corr_col_na_qtty) and (column not in self.undrop_cols):
                    # Remover `column` se tiver mais missings e não estiver em `undrop_cols`
                    to_drop.append(column)
                # tenta remover corr_col (ou tem mais missings ou column não é para ser removida)
                elif corr_col not in self.undrop_cols:
                    to_drop.append(corr_col)
                # corr_col não é para ser removida então tenta remover column
                elif column not in self.undrop_cols:
                    to_drop.append(column)

        # Remover colunas duplicadas na lista `to_drop`
        self.cols_to_drop_ = list(set(to_drop))
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_, errors='ignore')

class RemoveOutliersPercentilesTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers utilizando percentis (clip nos limites)."""
    def __init__(self, percentile=0.25):
        self.percentile = percentile

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        lower_percentile = self.percentile
        upper_percentile = 1 - lower_percentile
        self.lowers_ = {}
        self.uppers_ = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(lower_percentile)
            Q3 = X[col].quantile(upper_percentile)
            IQR = Q3 - Q1

            # Definir limites para outliers
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            
            self.lowers_[col] = lower_bound
            self.uppers_[col] = upper_bound
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()
        for col in self.lowers_.keys():
            X_out[col] = X_out[col].clip(lower=self.lowers_[col], upper=self.uppers_[col])
        return X_out

class TreatMissingValuesTransformer(BaseEstimator, TransformerMixin):
    """
    Trata valores ausentes (missing values) em colunas numéricas, booleanas e categóricas.
    
    Durante o fit, ele calcula, para cada coluna:
      - Para colunas booleanas: o valor mais frequente (moda).
      - Para colunas numéricas: a média.
      - Para colunas categóricas: a categoria mais frequente.
    
    Esses valores são armazenados e, no transform, são usados para preencher os NaNs.
    """
    
    def __init__(self):
        self.fill_values_ = {}
    
    def fit(self, X, y=None):
        # Garante que X seja um DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.fill_values_ = {}
        for col in X.columns:
            if pd.api.types.is_bool_dtype(X[col]):
                # Para colunas booleanas, preenche com a moda
                self.fill_values_[col] = X[col].mode()[0]
            elif pd.api.types.is_numeric_dtype(X[col]):
                # Para colunas numéricas, preenche com a média
                self.fill_values_[col] = X[col].median()
            elif pd.api.types.is_categorical_dtype(X[col]):
                # Para colunas categóricas, preenche com a moda
                self.fill_values_[col] = X[col].mode()[0]
            # Outras colunas não são tratadas, mas você pode incluir outras regras se necessário.
        return self

    def transform(self, X):
        # Garante que X seja um DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        for col, fill_val in self.fill_values_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(fill_val)
        return X_transformed

class DataFrameMinMaxScaler(BaseEstimator, TransformerMixin):
    """Aplica MinMaxScaler e retorna um DataFrame com as mesmas colunas e índice."""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.numeric_cols = None

    def fit(self, X, y=None):
        # Certifica-se de que X é DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Identifica as colunas numéricas
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Ajusta o scaler apenas nas colunas numéricas
        self.scaler.fit(X[self.numeric_cols]) 

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Cria uma cópia para evitar modificações no DataFrame original
        X_scaled = X.copy()

        # Aplica o scaler apenas nas colunas numéricas
        X_scaled[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        return X_scaled

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

# preprocessing_pipeline = Pipeline(steps=[
#     ('drop_columns', DropColumnsTransformer(columns_to_drop=['ID'])),
#     ('drop_single_value', DropSingleValueColumnsTransformer()),
#     ('replace_not_done', ReplaceTransformer(value_to_replace='Not done', replacement=np.nan)),
#     ('fix_bool', FixBoolTypeTransformer()),
#     ('convert_to_categorical', ConvertObjectToCategoricalTransformer(columns_to_exclude=[])),
#     ('ohe', OneHotEncodeCategoricalTransformer()),
#     ('drop_high_corr', DropHighCorrelationColumnsTransformer(threshold=0.95, undrop_cols=[])),
#     ('remove_outliers', RemoveOutliersPercentilesTransformer(percentile=0.25)),
#     ('treat_missing', TreatMissingValuesTransformer()),
#     ('scaler', DataFrameMinMaxScaler())
# ])

