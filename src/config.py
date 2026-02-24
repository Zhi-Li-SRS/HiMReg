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

        registration = self.config.setdefault("registration", {})
        registration.setdefault("register_type", "diff")
        registration.setdefault("wsireg_aligned_preset", False)

        scale_sync = self.config.setdefault("scale_sync", {})
        scale_sync.setdefault("enabled", True)
        scale_sync.setdefault("mode", "isotropic_fit")

        affine = self.config.setdefault("affine", {})
        if "scale_dependent_lr" not in affine and "scales" in affine:
            affine["scale_dependent_lr"] = [1e-4] * len(affine["scales"])
        affine.setdefault("patience", 50)
        affine.setdefault("min_delta", 1e-5)
        affine.setdefault("stages", ["rigid", "affine"])
        affine.setdefault("stage_iterations", {})
        affine.setdefault("loss_weights", {"mi": 1.0, "gradcc": 0.25})
        affine.setdefault("mask_weighted_loss", False)
        affine.setdefault("gradcc_sigma", 1.0)
        affine.setdefault("mi_num_samples", 15000)
        affine.setdefault("stage_mi_bins", {"rigid": 16, "affine": 32})
        affine.setdefault("scale_dependent_patience", True)
        affine.setdefault("center_mode", "none")
        affine.setdefault("required_valid_ratio", 0.05)
        affine.setdefault("invalid_sample_strategy", "lr_decay")
        affine.setdefault("invalid_lr_decay", 0.5)
        affine.setdefault("oob_penalty_weight", 0.0)
        affine.setdefault("oob_penalty_adaptive", True)
        # New affine defaults
        affine.setdefault("optimizer_type", "adam")
        affine.setdefault("constrained_affine", False)
        affine.setdefault("scale_mi_bins", {})
        affine.setdefault("scale_loss_schedule", {})
        affine.setdefault("asgd_a", None)
        affine.setdefault("asgd_alpha", 1.0)
        affine.setdefault("max_step_lengths", None)

        diff = self.config.setdefault("diff", {})
        diff.setdefault("tolerance", 5e-4)
        diff.setdefault("max_tolerance_iters", 80)
        diff.setdefault("optimizer_lr", 0.5)
        diff.setdefault("smooth_warp_sigma", 0.4)
        diff.setdefault("smooth_grad_sigma", 1.0)
        diff.setdefault("mi_num_samples", 25000)
        diff.setdefault("loss_weights", {"mi": 1.0, "gradcc": 0.25})
        diff.setdefault("gradcc_sigma", 1.0)
        diff.setdefault("mask_weighted_loss", False)
        diff.setdefault("regularization_weight", 0.05)

        # B-spline defaults (WSiReg 'nl' style)
        bspline = self.config.setdefault("bspline", {})
        bspline.setdefault("num_resolutions", 10)
        bspline.setdefault("final_grid_spacing", 100)
        bspline.setdefault("grid_spacing_schedule", [512, 392, 256, 128, 64, 32, 16, 4, 2, 1])
        bspline.setdefault("max_step_lengths", [100.0, 90.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 1.0, 1.0])
        bspline.setdefault("max_iterations", 200)
        bspline.setdefault("num_samples", 50000)
        bspline.setdefault("num_histogram_bins", 32)
        bspline.setdefault("bending_energy_weight", 0.0)
        bspline.setdefault("tolerance", 5e-4)
        bspline.setdefault("max_tolerance_iters", 80)
        bspline.setdefault("required_valid_ratio", 0.05)
        bspline.setdefault("invalid_sample_strategy", "lr_decay")
        bspline.setdefault("invalid_lr_decay", 0.5)
        bspline.setdefault("oob_penalty_weight", 0.0)
        bspline.setdefault("oob_penalty_adaptive", True)

        prepro = self.config.setdefault("preprocessing", {})
        prepro.setdefault(
            "fixed",
            {
                "image_type": "BF",
                "invert_bf": True,
                "gray_mode": "luma",
                "robust_normalize": True,
                "robust_percentiles": [1.0, 99.0],
            },
        )
        prepro.setdefault(
            "moving",
            {
                "image_type": "FL",
                "robust_normalize": True,
                "robust_percentiles": [1.0, 99.0],
            },
        )

        self._apply_wsireg_aligned_preset()

    def _apply_wsireg_aligned_preset(self) -> None:
        registration = self.config.setdefault("registration", {})
        if not bool(registration.get("wsireg_aligned_preset", False)):
            return

        affine = self.config.setdefault("affine", {})
        affine["loss_type"] = "mi"
        affine["optimizer_type"] = "asgd"
        affine["constrained_affine"] = False
        affine["asgd_a"] = None
        affine["asgd_alpha"] = 1.0
        affine["center_mode"] = "none"
        affine["scale_loss_schedule"] = {}
        affine["scale_mi_bins"] = {}

        bspline = self.config.setdefault("bspline", {})
        # Keep wsireg-like structure, but use stricter defaults in HiMReg to
        # avoid local-collapse artifacts in nonlinear optimization.
        bspline["required_valid_ratio"] = max(float(bspline.get("required_valid_ratio", 0.0)), 0.2)
        if str(bspline.get("invalid_sample_strategy", "none")).lower() == "none":
            bspline["invalid_sample_strategy"] = "lr_decay"
        bspline["oob_penalty_weight"] = max(float(bspline.get("oob_penalty_weight", 0.0)), 0.2)
        bspline["oob_penalty_adaptive"] = True

        scale_sync = self.config.setdefault("scale_sync", {})
        scale_sync["enabled"] = True
        scale_sync.setdefault("mode", "isotropic_fit")

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

        # Validate optimizer_type
        valid_optimizer_types = {"adam", "asgd"}
        if affine.get("optimizer_type", "adam") not in valid_optimizer_types:
            raise ValueError(f"Affine optimizer_type must be one of {sorted(valid_optimizer_types)}")

        # Validate scale_loss_schedule values
        valid_loss_types = ["mi", "cc", "dice", "mi_gradcc"]
        scale_loss_schedule = affine.get("scale_loss_schedule", {})
        if scale_loss_schedule:
            for tier, lt in scale_loss_schedule.items():
                if lt not in valid_loss_types:
                    raise ValueError(f"Invalid loss type '{lt}' in scale_loss_schedule['{tier}']")

        # Validate diff configuration
        diff = self.config["diff"]
        if len(diff["scales"]) != len(diff["iterations"]):
            raise ValueError("Diff scales and iterations must have the same length")
        if not isinstance(diff["tolerance"], (int, float)) or diff["tolerance"] < 0:
            raise ValueError("Diff tolerance must be non-negative")
        if not isinstance(diff["max_tolerance_iters"], int) or diff["max_tolerance_iters"] <= 0:
            raise ValueError("Diff max_tolerance_iters must be a positive integer")

        # Validate loss types
        if affine["loss_type"] not in valid_loss_types:
            raise ValueError(f"Invalid affine loss type: {affine['loss_type']}")

        if diff["loss_type"] not in valid_loss_types:
            raise ValueError(f"Invalid diff loss type: {diff['loss_type']}")

        if not isinstance(diff.get("mask_weighted_loss", True), bool):
            raise ValueError("Diff mask_weighted_loss must be boolean")
        if not isinstance(diff.get("regularization_weight", 0.05), (int, float)) or diff["regularization_weight"] < 0:
            raise ValueError("Diff regularization_weight must be non-negative")

        if not isinstance(affine["stages"], list) or len(affine["stages"]) == 0:
            raise ValueError("Affine stages must be a non-empty list")
        for stage_name in affine["stages"]:
            if stage_name not in {"rigid", "affine"}:
                raise ValueError(f"Unsupported affine stage: {stage_name}")

        if affine.get("center_mode", "image") not in {"none", "image", "tissue"}:
            raise ValueError("Affine center_mode must be one of: none, image, tissue")
        if not isinstance(affine.get("required_valid_ratio", 0.05), (int, float)) or affine["required_valid_ratio"] < 0:
            raise ValueError("Affine required_valid_ratio must be non-negative")
        if affine.get("invalid_sample_strategy", "lr_decay") not in {"none", "lr_decay", "early_stop"}:
            raise ValueError("Affine invalid_sample_strategy must be one of: none, lr_decay, early_stop")
        if not isinstance(affine.get("invalid_lr_decay", 0.5), (int, float)) or affine["invalid_lr_decay"] <= 0:
            raise ValueError("Affine invalid_lr_decay must be > 0")
        if not isinstance(affine.get("oob_penalty_weight", 0.0), (int, float)) or affine["oob_penalty_weight"] < 0:
            raise ValueError("Affine oob_penalty_weight must be non-negative")
        if not isinstance(affine.get("oob_penalty_adaptive", True), bool):
            raise ValueError("Affine oob_penalty_adaptive must be boolean")

        stage_iterations = affine.get("stage_iterations", {})
        if not isinstance(stage_iterations, dict):
            raise ValueError("Affine stage_iterations must be a dictionary")
        for stage_name, iters in stage_iterations.items():
            if stage_name not in {"rigid", "affine"}:
                raise ValueError(f"Unsupported stage_iterations key: {stage_name}")
            if not isinstance(iters, list) or len(iters) != len(affine["scales"]):
                raise ValueError(f"stage_iterations['{stage_name}'] must match affine scales length")

        runtime = self.config["runtime"]
        if not isinstance(runtime["seed"], int):
            raise ValueError("Runtime seed must be an integer")
        if not isinstance(runtime["deterministic"], bool):
            raise ValueError("Runtime deterministic flag must be boolean")

        # Validate register type
        valid_register_types = ["affine", "diff", "bspline"]
        register_type = self.config["registration"]["register_type"]
        if register_type not in valid_register_types:
            raise ValueError(f"Invalid register type: {register_type}")
        if not isinstance(self.config["registration"].get("wsireg_aligned_preset", False), bool):
            raise ValueError("registration.wsireg_aligned_preset must be boolean")

        scale_sync = self.config.get("scale_sync", {})
        if not isinstance(scale_sync.get("enabled", True), bool):
            raise ValueError("scale_sync.enabled must be boolean")
        if scale_sync.get("mode", "isotropic_fit") not in {"none", "isotropic_fit"}:
            raise ValueError("scale_sync.mode must be one of: none, isotropic_fit")

        # Validate bspline configuration
        bspline = self.config.get("bspline", {})
        nr = bspline.get("num_resolutions", 10)
        if not isinstance(nr, int) or nr < 1:
            raise ValueError("B-spline num_resolutions must be a positive integer")
        gs_sched = bspline.get("grid_spacing_schedule", [])
        ms_sched = bspline.get("max_step_lengths", [])
        if gs_sched and len(gs_sched) != nr:
            raise ValueError("B-spline grid_spacing_schedule length must match num_resolutions")
        if ms_sched and len(ms_sched) != nr:
            raise ValueError("B-spline max_step_lengths length must match num_resolutions")
        if not isinstance(bspline.get("required_valid_ratio", 0.05), (int, float)) or bspline["required_valid_ratio"] < 0:
            raise ValueError("B-spline required_valid_ratio must be non-negative")
        if bspline.get("invalid_sample_strategy", "lr_decay") not in {"none", "lr_decay", "early_stop"}:
            raise ValueError("B-spline invalid_sample_strategy must be one of: none, lr_decay, early_stop")
        if not isinstance(bspline.get("invalid_lr_decay", 0.5), (int, float)) or bspline["invalid_lr_decay"] <= 0:
            raise ValueError("B-spline invalid_lr_decay must be > 0")
        if not isinstance(bspline.get("oob_penalty_weight", 0.0), (int, float)) or bspline["oob_penalty_weight"] < 0:
            raise ValueError("B-spline oob_penalty_weight must be non-negative")
        if not isinstance(bspline.get("oob_penalty_adaptive", True), bool):
            raise ValueError("B-spline oob_penalty_adaptive must be boolean")

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
            "stages": affine.get("stages", ["rigid", "affine"]),
            "stage_iterations": affine.get("stage_iterations", {}),
            "loss_weights": affine.get("loss_weights", {"mi": 1.0, "gradcc": 0.25}),
            "mask_weighted_loss": affine.get("mask_weighted_loss", False),
            "gradcc_sigma": affine.get("gradcc_sigma", 1.0),
            "mi_num_samples": affine.get("mi_num_samples", 15000),
            "stage_mi_bins": affine.get("stage_mi_bins", {"rigid": 16, "affine": 32}),
            "scale_dependent_patience": affine.get("scale_dependent_patience", True),
            "center_mode": affine.get("center_mode", "none"),
            "required_valid_ratio": affine.get("required_valid_ratio", 0.05),
            "invalid_sample_strategy": affine.get("invalid_sample_strategy", "lr_decay"),
            "invalid_lr_decay": affine.get("invalid_lr_decay", 0.5),
            "oob_penalty_weight": affine.get("oob_penalty_weight", 0.0),
            "oob_penalty_adaptive": affine.get("oob_penalty_adaptive", True),
            # New fields
            "optimizer_type": affine.get("optimizer_type", "adam"),
            "constrained_affine": affine.get("constrained_affine", False),
            "scale_mi_bins": affine.get("scale_mi_bins", {}),
            "scale_loss_schedule": affine.get("scale_loss_schedule", {}),
            "asgd_a": affine.get("asgd_a"),
            "asgd_alpha": affine.get("asgd_alpha", 1.0),
            "max_step_lengths": affine.get("max_step_lengths"),
            "stage_lrs": affine.get("stage_lrs"),
            "optimizer_scales_from_physical_shift": affine.get("optimizer_scales_from_physical_shift", True),
        }

    def get_diff_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for diffeomorphic registration."""
        diff = self.config["diff"]
        return {
            "loss_type": diff["loss_type"],
            "tolerance": diff["tolerance"],
            "max_tolerance_iters": diff["max_tolerance_iters"],
            "optimizer_lr": diff.get("optimizer_lr", 0.5),
            "smooth_warp_sigma": diff.get("smooth_warp_sigma", 0.4),
            "smooth_grad_sigma": diff.get("smooth_grad_sigma", 1.0),
            "mi_num_samples": diff.get("mi_num_samples", 25000),
            "loss_weights": diff.get("loss_weights", {"mi": 1.0, "gradcc": 0.25}),
            "gradcc_sigma": diff.get("gradcc_sigma", 1.0),
            "mask_weighted_loss": diff.get("mask_weighted_loss", False),
            "regularization_weight": diff.get("regularization_weight", 0.05),
        }

    def get_bspline_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for B-spline registration (WSiReg-style)."""
        bs = self.config.get("bspline", {})
        return {
            "num_resolutions": bs.get("num_resolutions", 10),
            "final_grid_spacing": bs.get("final_grid_spacing", 100),
            "grid_spacing_schedule": bs.get("grid_spacing_schedule", [512, 392, 256, 128, 64, 32, 16, 4, 2, 1]),
            "max_step_lengths": bs.get("max_step_lengths", [100.0, 90.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 1.0, 1.0]),
            "max_iterations": bs.get("max_iterations", 200),
            "num_samples": bs.get("num_samples", 50000),
            "num_histogram_bins": bs.get("num_histogram_bins", 32),
            "bending_energy_weight": bs.get("bending_energy_weight", 0.0),
            "tolerance": bs.get("tolerance", 5e-4),
            "max_tolerance_iters": bs.get("max_tolerance_iters", 80),
            "required_valid_ratio": bs.get("required_valid_ratio", 0.05),
            "invalid_sample_strategy": bs.get("invalid_sample_strategy", "lr_decay"),
            "invalid_lr_decay": bs.get("invalid_lr_decay", 0.5),
            "oob_penalty_weight": bs.get("oob_penalty_weight", 0.0),
            "oob_penalty_adaptive": bs.get("oob_penalty_adaptive", True),
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
    def scale_sync_config(self) -> Dict[str, Any]:
        return self.config.get("scale_sync", {"enabled": True, "mode": "isotropic_fit"})

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

    @property
    def bspline_config(self) -> Dict[str, Any]:
        return self.config.get("bspline", {})

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        return self.config.get("preprocessing", {})


def load_config(yaml_path: str) -> ConfigManager:
    """Load configuration from YAML file with validation."""
    manager = ConfigManager.from_yaml(yaml_path)
    manager.validate()
    return manager
