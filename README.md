# HiMReg: Hierarchical Multimodal Image Registration Framework

HiMReg is a cross domain hierarchical multimodal registration framework with CUDA. It is specifically designed for co-registration for H&E and SRS, then align with MALDI for the metabolic and lipidomic correlation analysis.

## HiMReg Architecture
![Architecture](https://github.com/Zhi-Li-SRS/HiMReg/blob/main/flow_figure/flowchart.png?raw=true)

## Features
- **Hierarchical Registration**: Implements hierarchical registration with customizable scales and iterations
- **Multiple Registration Methods**:
  - Affine registration for global alignment
  - Diffeomorphic registration for non-linear vector field deformations
- **Loss Functions**:
  - Mutual Information (MI)
  - Local Normalized Cross Correlation (LNCC)
  - Support for any customized loss functions
- **Performance Optimization**:
  - GPU acceleration support
  - Batch processing capabilities
- **Evaluation Tools**:
  - Built-in comparison with Elastix registration
  - Multiple evaluation metrics (MI, DICE)
  - Visualization utilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HiMReg.git
cd HiMReg
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Registration

```python
from HiMReg import HiMReg
from data_load import Image

# Load images
fixed_image = Image.load_file("path/to/fixed_image.tif", device="cuda")
moving_image = Image.load_file("path/to/moving_image.tif", device="cuda")

# Initialize registration
registration = HiMReg(
    fixed_images=fixed_image,
    moving_images=moving_image,
    affine_scales=[8, 4, 2, 1],
    affine_iterations=[600, 400, 200, 100],
    diff_scales=[8, 4, 2, 1],
    diff_iterations=[600, 400, 200, 100]
)

# Perform registration
affine_transformed, diff_transformed, final_coordinates = registration.register(
    register_type="diff",
    save_transformed=True
)
```

### Command Line Interface

The framework also provides command-line tools for registration and comparison:

```bash
# Run registration
python HiMReg.py --fixed path/to/fixed.tif --moving path/to/moving.tif --output results

# Compare with Elastix
python compare.py --fixed path/to/fixed.tif --moving path/to/moving.tif --output comparison_results
```

## Technical Details

### Registration Pipeline

1. **Image Loading and Preprocessing**
   - Supports various image formats
   - Automatic intensity normalization
   - Multi-scale image pyramid generation

2. **Affine Registration**
   - Global transformation estimation
   - Optimization using Adam optimizer
   - MI or LNCC loss functions

3. **Diffeomorphic Registration**
   - deformation field estimation
   - Scaling and squaring for diffeomorphic transformation
   - Regularization for smooth deformations

4. **Evaluation**
   - Multiple similarity metrics
   - Comparison with traditional methods
   - Visualization tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{paper,
  title={},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```