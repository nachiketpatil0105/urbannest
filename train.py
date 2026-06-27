from utils.preprocess import load_and_encode
from utils.tune import run_tuning
from utils.evaluate import plot_convergence, plot_optuna_history, train_and_save_best

def run():
    print("Step 1: Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, _ = load_and_encode()
    print("\nStep 2: Running hyperparameter tuning...")
    results, study = run_tuning(X_train, y_train)
    print("\nStep 3: Generating plots...")
    plot_convergence(results)
    plot_optuna_history(study)
    print("\nStep 4: Training final model and saving...")
    train_and_save_best(results, X_train, y_train, X_test, y_test)
    print("\nTraining complete.")

if __name__ == "__main__":
    run()