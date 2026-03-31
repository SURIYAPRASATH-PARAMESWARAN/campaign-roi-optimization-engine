import logging, warnings
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from bank_roi.config import cfg
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
def _prep(X, ordinal=False):
    cat = X.select_dtypes(include=["object","string"]).columns.tolist()
    num = X.select_dtypes(exclude=["object","string"]).columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1) if ordinal else OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    return ColumnTransformer([("cat",enc,cat),("num",StandardScaler(),num)],verbose_feature_names_out=False)
def _obj(trial, name, X, y):
    c = cfg["optuna"]
    if name=="logistic_regression":
        m=LogisticRegression(C=trial.suggest_float("C",1e-3,10,log=True),solver=trial.suggest_categorical("solver",["lbfgs","saga"]),max_iter=5000,class_weight="balanced"); o=False
    elif name=="random_forest":
        m=RandomForestClassifier(n_estimators=trial.suggest_int("n",200,800,step=100),max_depth=trial.suggest_int("d",5,20),min_samples_leaf=trial.suggest_int("l",5,50),class_weight="balanced",random_state=42,n_jobs=-1); o=False
    elif name=="xgboost":
        m=XGBClassifier(n_estimators=trial.suggest_int("n",200,800,step=100),learning_rate=trial.suggest_float("lr",0.01,0.3,log=True),max_depth=trial.suggest_int("d",3,10),subsample=trial.suggest_float("ss",0.5,1.0),colsample_bytree=trial.suggest_float("cs",0.5,1.0),scale_pos_weight=trial.suggest_float("sp",5,15),verbosity=0,random_state=42,n_jobs=-1); o=False
    elif name=="lightgbm":
        m=LGBMClassifier(n_estimators=trial.suggest_int("n",200,800,step=100),learning_rate=trial.suggest_float("lr",0.01,0.3,log=True),num_leaves=trial.suggest_int("nl",20,150),max_depth=trial.suggest_int("d",3,12),min_child_samples=trial.suggest_int("mc",10,100),subsample=trial.suggest_float("ss",0.5,1.0),colsample_bytree=trial.suggest_float("cs",0.5,1.0),class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1); o=True
    else: raise ValueError(name)
    pipe=Pipeline([("p",_prep(X,o)),("m",m)])
    cv=StratifiedKFold(n_splits=c["cv_splits"],shuffle=True,random_state=c["random_state"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(pipe,X,y,cv=cv,scoring="average_precision",n_jobs=1)))
def tune_model(name, X, y, n_trials=None, timeout=None):
    c=cfg["optuna"]; n_trials=n_trials or c["n_trials"]; timeout=timeout or c["timeout"]
    study=optuna.create_study(direction=c["direction"],sampler=optuna.samplers.TPESampler(seed=c["random_state"]),study_name=name+"_tuning")
    study.optimize(lambda t:_obj(t,name,X,y),n_trials=n_trials,timeout=timeout,show_progress_bar=True,n_jobs=1)
    logger.info("%s best PR-AUC: %.4f",name,study.best_value)
    return {"model_name":name,"best_params":study.best_params,"best_value":study.best_value,"study":study,"trials_df":study.trials_dataframe()}
def tune_all_models(X, y, models=None):
    if models is None: models=["logistic_regression","random_forest","xgboost","lightgbm"]
    return {n:tune_model(n,X,y) for n in models}
