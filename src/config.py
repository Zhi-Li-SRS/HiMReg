import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """Simple configuration manager for HiMReg."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        self.config = config_dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ConfigManager':
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        return cls(yaml_data)
    
    def validate(self):
        """Validate configuration parameters."""
        affine = self.config['affine']
        if len(affine['scales']) != len(affine['iterations']):
            raise ValueError("Affine scales and iterations must have the same length")
        
        if len(affine['scales']) != len(affine['scale_dependent_lr']):
            raise ValueError("Affine scales and scale_dependent_lr must have the same length")
        
        # Validate diff configuration
        diff = self.config['diff']
        if len(diff['scales']) != len(diff['iterations']):
            raise ValueError("Diff scales and iterations must have the same length")
        
        # Validate loss types
        valid_loss_types = ["mi", "cc", "dice"]
        if affine['loss_type'] not in valid_loss_types:
            raise ValueError(f"Invalid affine loss type: {affine['loss_type']}")
        
        if diff['loss_type'] not in valid_loss_types:
            raise ValueError(f"Invalid diff loss type: {diff['loss_type']}")
        
        # Validate register type
        valid_register_types = ["affine", "diff"]
        register_type = self.config['registration']['register_type']
        if register_type not in valid_register_types:
            raise ValueError(f"Invalid register type: {register_type}")
        
        # Validate file paths
        io_config = self.config['io']
        if not os.path.exists(io_config['fixed']):
            raise FileNotFoundError(f"Fixed image not found: {io_config['fixed']}")
        
        if not os.path.exists(io_config['moving']):
            raise FileNotFoundError(f"Moving image not found: {io_config['moving']}")
    
    def get_affine_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for affine registration."""
        return {"loss_type": self.config['affine']['loss_type']}
    
    def get_diff_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for diffeomorphic registration."""
        return {"loss_type": self.config['diff']['loss_type']}
    
    @property
    def fixed_image_path(self) -> str:
        return self.config['io']['fixed']
    
    @property
    def moving_image_path(self) -> str:
        return self.config['io']['moving']
    
    @property
    def output_dir(self) -> str:
        return self.config['io']['output']
    
    @property
    def register_type(self) -> str:
        return self.config['registration']['register_type']


def load_config(yaml_path: str) -> ConfigManager:
    """Load configuration from YAML file with validation."""
    manager = ConfigManager.from_yaml(yaml_path)
    manager.validate()
    return manager