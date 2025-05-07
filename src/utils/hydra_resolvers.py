from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

def get_compact_override_string():
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        return "new"

    short_tags = []
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=")
        parts = key.split(".")
        abbrev = "-".join([p[:2] for p in parts])
        short_tags.append(f"{abbrev}{value}")
    return "_".join(short_tags) or "new"

if not OmegaConf.has_resolver("override_name"):
    OmegaConf.register_new_resolver("override_name", get_compact_override_string)

