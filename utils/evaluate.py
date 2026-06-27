import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from optuna.visualization.matplotlib import plot_optimization_history

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

def plot_convergence(results):
    PLOTS_DIR.mkdir(exist_ok=True)
    
    styles = {
        "Grid Search": ("steelblue", "--", "s"),
        "Random Search": ("seagreen", "-.", "^"),
        "Bayesian (Optuna)": ("tomato", "-", "o"),
    }
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for method, (mae, params, t, curve) in results.items():
        color, ls, marker = styles[method]
        iterations = range(1, len(curve) + 1)
        
        ax.plot(iterations, curve, label=method, color=color, linestyle=ls, marker=marker, markersize=4, linewidth=2 if method == "Bayesian (Optuna)" else 1)
                
    ax.set_title("Hyperparameter Optimization — Convergence Comparison")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Best CV MAE So Far (Rs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    fig.savefig(PLOTS_DIR / "trials_vs_error.png", dpi=150)
    plt.show()
    print("Saved plots/trials_vs_error.png")

def plot_optuna_history(study):
    PLOTS_DIR.mkdir(exist_ok=True)
    
    ax = plot_optimization_history(study)
    fig = ax.get_figure()
    fig.set_size_inches(12, 5)
    
    ax.set_title("Optuna Optimization History")
    fig.tight_layout()
    
    fig.savefig(PLOTS_DIR / "optuna_hyperparameter_space.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved plots/optuna_hyperparameter_space.png")

def train_and_save_best(results, X_train, y_train, X_test, y_test):
    print(f"\n{'Method':<22} {'CV MAE':>12}  {'Time':>8}")
    print("-" * 46)
    
    for method, (mae, params, t, _) in results.items():
        print(f"{method:<22} Rs {mae:>9,.0f}  {t:>7.1f}s")
        print(f"  {params}")
        
    best_method = min(results, key=lambda k: results[k][0])
    best_params = results[best_method][1]
    print(f"\nBest method: {best_method}  (Rs {results[best_method][0]:,.0f} CV MAE)")
    
    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    
    if y_test is not None:
        test_mae = mean_absolute_error(y_test, final_model.predict(X_test))
        print(f"Test MAE: Rs {test_mae:,.0f}")
    else:
        print("Test MAE: N/A (Blind test set without target column)")
        
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "best_rf_model.pkl")
    print("Saved models/best_rf_model.pkl")
    
    return final_model

if __name__ == "__main__":
    import sys
    
    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    
    from utils.preprocess import load_and_encode
    from utils.tune import run_tuning
    
    X_train, y_train, X_test, y_test, _ = load_and_encode()
    results, study = run_tuning(X_train, y_train)
    
    plot_convergence(results)
    plot_optuna_history(study)
    train_and_save_best(results, X_train, y_train, X_test, y_test)