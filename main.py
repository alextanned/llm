import hydra
from config.algo import AlgoConfig


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg):
    from train_rl import train
    from omegaconf import OmegaConf
    algo_cfg = OmegaConf.to_object(cfg)
    assert isinstance(algo_cfg, AlgoConfig)
    train(algo_cfg)


if __name__ == "__main__":
    main()
