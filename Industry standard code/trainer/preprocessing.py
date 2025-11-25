
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow

def build_preprocessor(df,scaling=True):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["float64","int64"]).columns

    transformers= []
    if scaling:
        transformers.append(("num", StandardScaler(), numerical_cols))
    transformers.append(("cat",OneHotEncoder(handle_unknown="ignore"),categorical_cols))
    processor = ColumnTransformer(transformers)

    # mlflow.log_param("target_encoder_clases",list(transformers.classes_))

    return processor