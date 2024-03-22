import optuna
import traceback
from omegaconf import OmegaConf
from hydra import compose, initialize
from rnnt.train import train

def objective(trial):
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.0, 100.0)
    rnnt_grad_clamp = trial.suggest_float("rnnt_grad_clamp", 0.0, 100.0)

    with initialize(config_path="config"):
        # Load your existing configuration
        cfg = compose(config_name="basic_sp_conv.yaml")
        
        # Update the configuration with the suggested hyperparameters
        cfg.training.clip_grad_norm = clip_grad_norm
        cfg.training.rnnt_grad_clamp = rnnt_grad_clamp

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


    study = optuna.create_study(study_name="grad_clamping_study", 
                                direction="minimize",
                                storage= optuna.storages.RDBStorage(url="postgresql://optuna_user:password@localhost/optuna_db"),
                                load_if_exists=True,
                                )

    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=10)
