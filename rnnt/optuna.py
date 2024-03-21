import optuna
import traceback
from omegaconf import OmegaConf
from hydra import compose, initialize
from rnnt.train import train

def objective(trial):
    beta0_inv = trial.suggest_float("beta0_inv", 1e-6, 0.2, log=True)
    beta1_inv = trial.suggest_float("beta1_inv", 1e-6, 1.0, log=True)

    # beta1_inv == 0.0 means beta1 = 1.0
    # beta1_inv == 0.5, beta1 = halfway between beta0 and 1.0, etc.
    beta0 = 1.0 - beta0_inv
    beta1 = beta0 + (1.0 - beta0) * (1.0 - beta1_inv)

    with initialize(config_path="config"):
        # Load your existing configuration
        cfg = compose(config_name="basic_sp_conv.yaml")
        
        # Update the configuration with the suggested hyperparameters
        cfg.training.optimizer.betas = [beta0, beta1]

        # Run training with the current set of hyperparameters
        try:
            wer = train(cfg)
        except Exception as e:
            print(f"Exception during training: {e}")
            print(traceback.format_exc())
            trial.set_user_attr("exception", str(e))
            wer = float('inf')

    # Optuna aims to minimize the objective, so if you have a metric that should be maximized, 
    # you need to return its negative value. Here, we assume WER (Word Error Rate) should be minimized.
    return wer


if __name__ == "__main__":


    study = optuna.create_study(study_name="single_epoch_conv_beta_study", 
                                direction="minimize",
                                storage= optuna.storages.RDBStorage(url="postgresql://optuna_user:password@localhost/optuna_db"),
                                load_if_exists=True,
                                )

    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=10)
