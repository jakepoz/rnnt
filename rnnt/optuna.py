import optuna
import traceback
from omegaconf import OmegaConf
from hydra import compose, initialize
from rnnt.train import train

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    norm_type = trial.suggest_categorical("norm_type", ["batch", "instance", "instance_affine"])


    with initialize(config_path="config"):
        # Load your existing configuration
        cfg = compose(config_name="basic_sp.yaml")
        
        # Update the configuration with the suggested hyperparameters
        cfg.training.optimizer.lr = learning_rate
        cfg.training.pergpu_minibatch_size = batch_size
        cfg.encoder.norm_type = norm_type

        for block in cfg.encoder.blocks:
            block.norm_type = norm_type

        # Run training with the current set of hyperparameters
        try:
            wer = train(cfg)
        except Exception as e:
            print(f"Exception during training: {e}")
            print(traceback.format_exc())
            wer = float('inf')

    # Optuna aims to minimize the objective, so if you have a metric that should be maximized, 
    # you need to return its negative value. Here, we assume WER (Word Error Rate) should be minimized.
    return wer


if __name__ == "__main__":


    study = optuna.create_study(study_name="single_epoch_batch_and_norms", 
                                direction="minimize",
                                storage= optuna.storages.RDBStorage(url="postgresql://optuna_user:password@localhost/optuna_db"),
                                load_if_exists=True,
                                )

    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=10)
