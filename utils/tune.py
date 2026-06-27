import sys
import time
import optuna
import numpy as np
import trackio
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.preprocess import load_and_encode

def run_tuning(X_train, y_train):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rf_base = RandomForestRegressor(random_state=42)
    results = {}

    # ---------------------------------------------------------
    # 1. Grid Search
    # ---------------------------------------------------------
    trackio.init(project="UrbanNest", name="tuning_GridSearch", resume="never")
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [10, 15, 20, 25, 30],
        "min_samples_split": [2, 5, 8],
    }
    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - t0
    
    grid_best_score = -grid_search.best_score_
    grid_curve, best = [], float("inf")
    for s in grid_search.cv_results_["mean_test_score"]:
        best = min(best, -s)
        grid_curve.append(best)
        
    trackio.log({
        "method": "GridSearch", 
        "time_taken": float(grid_time),
        "num_iterations": len(grid_search.cv_results_["params"]),
        "best_cv_mae": float(grid_best_score),
        **{f"best_{k}": int(v) for k, v in grid_search.best_params_.items()}
    })
    trackio.finish()
    
    print(f"Grid Search  |  {grid_time:.1f}s  |  CV MAE: Rs {grid_best_score:,.0f}  |  {grid_search.best_params_}")
    results["Grid Search"] = (grid_best_score, grid_search.best_params_, grid_time, grid_curve)


    # ---------------------------------------------------------
    # 2. Random Search
    # ---------------------------------------------------------
    trackio.init(project="UrbanNest", name="tuning_RandomSearch", resume="never")
    param_dist = {
        "n_estimators": np.arange(50, 201),
        "max_depth": np.arange(10, 31),
        "min_samples_split": np.arange(2, 11),
    }
    random_search = RandomizedSearchCV(rf_base, param_dist, n_iter=60, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
    
    t0 = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - t0
    
    random_best_score = -random_search.best_score_
    random_curve, best = [], float("inf")
    for s in random_search.cv_results_["mean_test_score"]:
        best = min(best, -s)
        random_curve.append(best)
        
    trackio.log({
        "method": "RandomSearch", 
        "time_taken": float(random_time),
        "num_iterations": len(random_search.cv_results_["params"]),
        "best_cv_mae": float(random_best_score),
        **{f"best_{k}": int(v) for k, v in random_search.best_params_.items()}
    })
    trackio.finish() # Close the run
    
    print(f"Random Search  |  {random_time:.1f}s  |  CV MAE: Rs {random_best_score:,.0f}  |  {random_search.best_params_}")
    results["Random Search"] = (random_best_score, random_search.best_params_, random_time, random_curve)


    # ---------------------------------------------------------
    # 3. Bayesian Optimization (Optuna)
    # ---------------------------------------------------------
    trackio.init(project="UrbanNest", name="tuning_Bayesian", resume="never")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
        mdl = RandomForestRegressor(**params, random_state=42)
        scores = cross_val_score(mdl, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        return -scores.mean()
        
    study = optuna.create_study(direction="minimize")
    
    t0 = time.time()
    study.optimize(objective, n_trials=60)
    bayesian_time = time.time() - t0
    
    bayesian_best_score = study.best_value
    bayesian_curve, current_best = [], float("inf")
    for t in study.trials:
        if t.value is not None:
            current_best = min(current_best, t.value)
            bayesian_curve.append(current_best)
            
    trackio.log({
        "method": "BayesianOptimization", 
        "time_taken": float(bayesian_time),
        "num_iterations": len(study.trials), 
        "best_cv_mae": float(bayesian_best_score),
        **{f"best_{k}": int(v) for k, v in study.best_params.items()}
    })
    trackio.finish()
    
    print(f"Bayesian  |  {bayesian_time:.1f}s  |  CV MAE: Rs {bayesian_best_score:,.0f}  |  {study.best_params}")
    results["Bayesian (Optuna)"] = (bayesian_best_score, study.best_params, bayesian_time, bayesian_curve)

    return results, study

if __name__ == "__main__":
    X_train, y_train, _, _, _ = load_and_encode()
    run_tuning(X_train, y_train)