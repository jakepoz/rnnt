import optuna
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from rnnt.train import train

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    beta0 = trial.suggest_float("beta0", 0.9, 0.999)
    beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    apply_linear_log = trial.suggest_categorical("apply_linear_log", [True, False])
    featurize_mean = trial.suggest_float("featurize_mean", 0.0, 30.0)
    featurize_invstddev = trial.suggest_float("featurize_invstddev", 0.1, 0.5)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 10, log=True)
    optimizer_eps = trial.suggest_float("optimzier_eps", 1e-9, 1e-5, log=True)
    
    with initialize(config_path="config"):
        # Load your existing configuration
        cfg = compose(config_name="basic_sp.yaml")
        
        # Update the configuration with the suggested hyperparameters
        cfg.training.optimizer.lr = learning_rate
        cfg.training.optimizer.betas = (beta0, beta1)
        cfg.training.optimizer.weight_decay = weight_decay
        cfg.training.optimizer.eps = optimizer_eps

        cfg.featurizer.apply_linear_log = apply_linear_log
        cfg.featurizer.mean = featurize_mean
        cfg.featurizer.invstddev = featurize_invstddev
        
        cfg.predictor.lstm.dropout = lstm_dropout
        
        # Run training with the current set of hyperparameters
        try:
            wer = train(cfg)
        except Exception as e:
            print(e)
            wer = float('inf')

    # Optuna aims to minimize the objective, so if you have a metric that should be maximized, 
    # you need to return its negative value. Here, we assume WER (Word Error Rate) should be minimized.
    return wer

# Define the study and start the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
