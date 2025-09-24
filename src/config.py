import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """Simple configuration manager for HiMReg."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        self.config = config_dict
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Populate optional configuration entries with sensible defaults."""
        runtime = self.config.setdefault("runtime", {})
        runtime.setdefault("seed", 42)
        runtime.setdefault("deterministic", True)

        affine = self.config.setdefault("affine", {})
        if "scale_dependent_lr" not in affine and "scales" in affine:
            affine["scale_dependent_lr"] = [1e-4] * len(affine["scales"])
        affine.setdefault("patience", 50)
        affine.setdefault("min_delta", 1e-5)

        diff = self.config.setdefault("diff", {})
        diff.setdefault("tolerance", 1e-3)
        diff.setdefault("max_tolerance_iters", 100)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ConfigManager":
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        return cls(yaml_data)

    def validate(self):
        """Validate configuration parameters."""
        affine = self.config["affine"]
        if len(affine["scales"]) != len(affine["iterations"]):
            raise ValueError("Affine scales and iterations must have the same length")

        if len(affine["scales"]) != len(affine["scale_dependent_lr"]):
            raise ValueError("Affine scales and scale_dependent_lr must have the same length")
        if not isinstance(affine["patience"], int) or affine["patience"] <= 0:
            raise ValueError("Affine patience must be a positive integer")
        if not isinstance(affine["min_delta"], (int, float)) or affine["min_delta"] < 0:
            raise ValueError("Affine min_delta must be non-negative")

        # Validate diff configuration
        diff = self.config["diff"]
        if len(diff["scales"]) != len(diff["iterations"]):
            raise ValueError("Diff scales and iterations must have the same length")
        if not isinstance(diff["tolerance"], (int, float)) or diff["tolerance"] < 0:
            raise ValueError("Diff tolerance must be non-negative")
        if not isinstance(diff["max_tolerance_iters"], int) or diff["max_tolerance_iters"] <= 0:
            raise ValueError("Diff max_tolerance_iters must be a positive integer")

        # Validate loss types
        valid_loss_types = ["mi", "cc", "dice"]
        if affine["loss_type"] not in valid_loss_types:
            raise ValueError(f"Invalid affine loss type: {affine['loss_type']}")

        if diff["loss_type"] not in valid_loss_types:
            raise ValueError(f"Invalid diff loss type: {diff['loss_type']}")

        runtime = self.config["runtime"]
        if not isinstance(runtime["seed"], int):
            raise ValueError("Runtime seed must be an integer")
        if not isinstance(runtime["deterministic"], bool):
            raise ValueError("Runtime deterministic flag must be boolean")

        # Validate register type
        valid_register_types = ["affine", "diff"]
        register_type = self.config["registration"]["register_type"]
        if register_type not in valid_register_types:
            raise ValueError(f"Invalid register type: {register_type}")

        # Validate file paths
        io_config = self.config["io"]
        if not os.path.exists(io_config["fixed"]):
            raise FileNotFoundError(f"Fixed image not found: {io_config['fixed']}")

        if not os.path.exists(io_config["moving"]):
            raise FileNotFoundError(f"Moving image not found: {io_config['moving']}")

    def get_affine_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for affine registration."""
        affine = self.config["affine"]
        return {
            "loss_type": affine["loss_type"],
            "patience": affine["patience"],
            "min_delta": affine["min_delta"],
        }

    def get_diff_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for diffeomorphic registration."""
        diff = self.config["diff"]
        return {
            "loss_type": diff["loss_type"],
            "tolerance": diff["tolerance"],
            "max_tolerance_iters": diff["max_tolerance_iters"],
        }

    @property
    def fixed_image_path(self) -> str:
        return self.config["io"]["fixed"]

    @property
    def moving_image_path(self) -> str:
        return self.config["io"]["moving"]

    @property
    def output_dir(self) -> str:
        return self.config["io"]["output"]

    @property
    def register_type(self) -> str:
        return self.config["registration"]["register_type"]

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.config["runtime"]

    @property
    def seed(self) -> int:
        return int(self.runtime["seed"])

    @property
    def deterministic(self) -> bool:
        return bool(self.runtime["deterministic"])

    @property
    def affine_config(self) -> Dict[str, Any]:
        return self.config["affine"]

    @property
    def diff_config(self) -> Dict[str, Any]:
        return self.config["diff"]


def load_config(yaml_path: str) -> ConfigManager:
    """Load configuration from YAML file with validation."""
    manager = ConfigManager.from_yaml(yaml_path)
    manager.validate()
    return manager
