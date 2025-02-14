import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class ParticipantVisibleError(Exception):
    pass


def calc_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = 'ID') -> float:
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

def createSubmission(X_test, y_pred):
    # Create the submission: ID, prediction
    submission = pd.DataFrame({
        'ID': X_test.ID, 
        'prediction': y_pred
    })
    return submission

def strip_str_columns(df):
    for col in df.columns:
        # Verifica se a coluna é predominantemente strings ou vazia
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Remove valores não string e aplica .str.strip() apenas em strings
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# Mostra as diferenças entre df1 e df2
def show_dif_dfs(df1, df2):
    # Usando merge para encontrar linhas únicas em cada DataFrame
    merged = df1.merge(df2, how='outer', indicator=True)

    # Filtrando as linhas que não estão em ambos os DataFrames
    unique_to_a = merged[merged['_merge'] == 'left_only']
    unique_to_b = merged[merged['_merge'] == 'right_only']

    # Resultados
    print("Linhas únicas em df1:")
    print(unique_to_a.drop(columns='_merge'))

    print("\nLinhas únicas em df2:")
    print(unique_to_b.drop(columns='_merge'))

true_values = ['sim', 's', '1.0', '1', 'verdadeiro', 'positivo', 'true', 'checked', 'yes', 'positive']
false_values = ['não', 'nao', 'n', '0.0', '0', 'falso', 'negativo', 'false', 'unchecked', 'no', 'negative']
bool_values = true_values + false_values

def col_is_bool(df, col):
    unique_values = df[col].dropna().astype(str).str.lower().unique()
    if len(unique_values) == 0:
        return False
    else:
        return set(unique_values).issubset(bool_values)

def convert_to_bool(value):
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
    elif isinstance(value, (int, float)):
        if value == 0:
            return False
        elif value == 1:
            return True
    return value

def fix_bol_type(df):
    for col in df.columns:
        if col_is_bool(df, col):
            df[col] = df[col].apply(convert_to_bool).astype('boolean')
            # df_col = df[col].apply(convert_to_bool)
            # df_col = df_col.astype('boolean')
            # df.loc[:,col] = df_col
    return df

def convert_object_columns_to_categorical(df: pd.DataFrame, columns_to_exclude: list) -> pd.DataFrame:
    """
    Converte colunas do tipo 'object' em um DataFrame para o tipo 'category',
    excluindo colunas especificadas.

    Args:
        df: O DataFrame do Pandas a ser modificado.
        columns_to_exclude: Uma lista de nomes de colunas a serem excluídas da conversão.

    Returns:
        O DataFrame modificado com as colunas especificadas convertidas para o tipo 'category'.
    """

    # Seleciona as colunas do DataFrame que são do tipo 'object'
    object_columns = df.select_dtypes(include='object')

    # Identifica as colunas que devem ser convertidas para 'category', excluindo aquelas
    # presentes na lista columns_to_exclude.
    columns_to_convert = [
        column for column in object_columns.columns if column not in columns_to_exclude
    ]

    # Itera sobre a lista de colunas a serem convertidas e realiza a conversão para 'category'.
    for column in columns_to_convert:
        df[column] = df[column].astype('category')

    return df

def convert_cols_to_categorical(df, columns_to_convert):
    """
    Converte colunas para o tipo 'category', garantindo que colunas numéricas
    ou de outros tipos sejam convertidas para string antes da conversão.

    Args:
        df: DataFrame Pandas a ser modificado.
        columns_to_convert: Lista de colunas para converter para 'category'.

    Returns:
        DataFrame Pandas com as colunas convertidas para 'category'.
    """

    for column in columns_to_convert:
        # Verifica se a coluna é do tipo numérico
        if pd.api.types.is_numeric_dtype(df[column]):
            # Se for numérica, converte para string antes de converter para 'category'
            df[column] = df[column].astype(str)

        # Agora que a coluna é string, pode ser convertida para 'category' com segurança
        df[column] = df[column].astype('category')

    return df

def ohe_catCols(df):
    """
    Aplica One-Hot Encoding (OHE) para colunas categóricas com mais de 2 categorias
    e Label Encoding (LE) para colunas categóricas com 2 categorias em um DataFrame.

    Args:
        df: O DataFrame do Pandas a ser processado.

    Returns:
        O DataFrame modificado com as colunas categóricas transformadas.
    """
    df = df.copy()  # Evita modificar o DataFrame original
    categorical_cols = df.select_dtypes(include='category').columns

    for col in categorical_cols:
        num_categorias = df[col].nunique()

        if num_categorias > 2:
            # One-Hot Encoding
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(encoded_data, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]], index=df.index)

            # Remove a coluna original e concatena as novas colunas OHE
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)

        elif num_categorias == 2:
            # Label Encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

def getTextCols(df):
    return df.select_dtypes(include=["object", "string"])

def ohe_cols(df, cols):
    # Instanciar o OneHotEncoder
    encoder = OneHotEncoder(drop="if_binary", sparse_output=False)  # Para retornar DataFrame ou matriz densa
    encoder.set_output(transform="pandas")

    # Aplicar o encoder somente nas colunas selecionadas
    encoded_array = encoder.fit_transform(df[cols])

    # Criar DataFrame das colunas codificadas
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(cols),
        index=df.index
    )

    # Combinar com o DataFrame original (removendo as colunas originais codificadas)
    df_encoded = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    return df_encoded

def ohe_cols_str(df):
    # Identificar as colunas do tipo string
    string_columns = df.select_dtypes(include=["object", "string"]).columns
    # Instanciar o OneHotEncoder
    encoder = OneHotEncoder(drop="if_binary", sparse_output=False)  # Para retornar DataFrame ou matriz densa
    encoder.set_output(transform="pandas")

    # Aplicar o encoder somente nas colunas string
    encoded_array = encoder.fit_transform(df[string_columns])

    # Criar DataFrame das colunas codificadas
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(string_columns),
        index=df.index
    )

    # Combinar com o DataFrame original (removendo as colunas originais codificadas)
    df_encoded = pd.concat([df.drop(columns=string_columns), encoded_df], axis=1)
    return df_encoded

def split_column_to_booleans(df, column, option1, option2, both_value='ambas'):
    """
    Divide uma coluna com duas opções e uma opção "ambas" em duas colunas booleanas.

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        column (str): Nome da coluna a ser processada.
        option1 (str): Primeira opção de valor na coluna.
        option2 (str): Segunda opção de valor na coluna.
        both_value (str): Valor que representa "ambas" (default: 'ambas').

    Retorna:
        pd.DataFrame: Novo DataFrame com as duas novas colunas booleanas.
    """
    new_df = df.copy()
    # Criar as novas colunas booleanas
    new_df[f'{column}_{option1}'] = new_df[column].apply(lambda x: x in [option1, both_value])
    new_df[f'{column}_{option2}'] = new_df[column].apply(lambda x: x in [option2, both_value])
    # Remover a coluna original
    new_df.drop(columns=[column], inplace=True)
    return new_df

# Função para remover colunas com alta correlação
def drop_high_corr_cols(df, threshold, undrop_cols):
    """
    Remove colunas altamente correlacionadas com base em um limite de correlação.
    Dá preferência para remover colunas com mais valores ausentes e preserva colunas específicas, se especificadas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        threshold (float): Limite de correlação acima do qual as colunas serão consideradas altamente correlacionadas.
        undrop_cols (list): Lista de colunas que não devem ser removidas.

    Returns:
        pd.DataFrame: DataFrame sem as colunas altamente correlacionadas.
    """

    # Calcular a matriz de correlação absoluta
    correlation_matrix = df.corr().abs()

    # Obter apenas o triângulo superior da matriz de correlação para evitar duplicatas
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Lista para armazenar colunas a serem removidas
    to_drop = []

    # Iterar pelas colunas no triângulo superior
    for column in upper_triangle.columns:
        # Encontrar colunas altamente correlacionadas com a coluna atual
        highly_corr_cols = upper_triangle.index[upper_triangle[column] > threshold].tolist()

        # Iterar pelas colunas correlacionadas
        for corr_col in highly_corr_cols:
            # Contar valores ausentes em ambas as colunas
            column_na_qtty = df[column].isna().sum()
            corr_col_na_qtty = df[corr_col].isna().sum()

            # Decidir qual coluna remover com base na quantidade de valores ausentes
            if (column_na_qtty > corr_col_na_qtty) and (column not in undrop_cols):
                # Remover `column` se tiver mais missings e não estiver em `undrop_cols`
                to_drop.append(column)
            # tenta remover corr_col (ou tem mais missings ou column não é para ser removida)
            elif corr_col not in undrop_cols:
                to_drop.append(corr_col)
            # corr_col não é para ser removida então tenta remover column
            elif column not in undrop_cols:
                to_drop.append(column)

    # Remover colunas duplicadas na lista `to_drop`
    to_drop = list(set(to_drop))

    # Retornar DataFrame sem as colunas a serem removidas
    return df.drop(columns=to_drop)

def drop_single_value_columns(df):
    """
    Remove colunas que possuem apenas um valor único.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: DataFrame sem colunas com um único valor.
    """
    return df.loc[:, df.nunique() > 1]

def remove_outliers_percentiles(df, percentile=0.25):
    """
    Remove outliers com base nos percentis inferior e superior.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        percentile (float): Percentil para corte.

    Returns:
        pd.DataFrame: DataFrame ajustado com outliers substituídos.
    """
    lower_percentile = percentile
    upper_percentile = 1 - lower_percentile

    df_adjusted = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(lower_percentile)
        Q3 = df[col].quantile(upper_percentile)
        IQR = Q3 - Q1

        # Definir limites para outliers
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        df_adjusted[col] = np.clip(df[col], lower_bound, upper_bound)
    return df_adjusted

def basic_treat_missing_numbers_booleans(df):
    """
    Trata valores ausentes (missing values) em um DataFrame:
    - Para colunas booleanas: preenche com o valor mais frequente (moda).
    - Para colunas numéricas: preenche com a média.
    - Para colunas categóricas: preenche com a categoria mais frequente.
    - Outras colunas não são alteradas.

    Args:
        df (pd.DataFrame): DataFrame de entrada com valores ausentes.

    Returns:
        pd.DataFrame: DataFrame com os valores ausentes tratados.
    """
    df_filled = df.copy()

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            # Preencher valores ausentes com a moda (valor mais frequente)
            mode_value = df[col].mode()[0]  # Moda retorna uma série; pega o primeiro elemento
            df_filled[col] = df_filled[col].fillna(mode_value)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Preencher valores ausentes com a média
            mean_value = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_value)
        elif pd.api.types.is_categorical_dtype(df[col]):
            mode_value = df[col].mode()[0]  # Moda retorna uma série; pega o primeiro elemento
            df_filled[col] = df_filled[col].fillna(mode_value)   

    return df_filled

