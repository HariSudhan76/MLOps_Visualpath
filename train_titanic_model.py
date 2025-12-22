import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os 
import json 

##Load dataset
df = sns.load_dataset('titanic')

##Select useful features - feature selection
features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
df = df[features+["survived"]]

#Simple cleaning
df["age"]= df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna("S")

X= df[features]
y = df["survived"]


##Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23, stratify=y)

#preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),["sex","embarked"]),
        ("num","passthrough",["pclass","age","sibsp","parch","fare"])
    ]
)

#Full pipeline
pipeline = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(n_estimators=100,random_state=23))
])

pipeline.fit(X_train,y_train)


#Evaluate
y_pred = pipeline.predict(X_test)
y_proba =pipeline.predict_proba(X_test)[:,1]
report = classification_report(y_test,y_pred, output_dict=True)
auc = float(roc_auc_score(y_test,y_proba))
cm = confusion_matrix(y_test,y_pred).tolist()


##Save model and metric values
os.makedirs("artifacts",exist_ok=True)
joblib.dump(pipeline,"artifacts/titanic_model.pkl")

with open("artifacts/metric.json","w") as f:
    json.dump({"classification_report":report, "roc_auc_score":auc, "confusion_matrix":cm},f, indent=2)

fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(confusion_matrix(y_test,y_pred), cmap="Blues", alpha=0.7)
for (i,j), z in np.ndenumerate(confusion_matrix(y_test,y_pred)):
    ax.text(j, i, str(z), ha="center",va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.title(f"Confusion matrix (AUC={auc:.3f})")
plt.savefig("artifacts/confusion_matrix.png",bbox_inches="tight")
plt.close()