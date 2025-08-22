#!/bin/bash

# Neural Mesh Simplification - Installation Script for Ubuntu 24 EC2
# This script sets up all dependencies required for training the neural mesh simplification model
# Designed for Ubuntu 24 EC2 instances with GPU driver AMI (CUDA 12.1)

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check if running as root (not recommended for venv)
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons."
   log_error "Please run as a regular user with sudo privileges."
   exit 1
fi

# Script header
echo "================================================================="
echo "Neural Mesh Simplification - Dependency Installation Script"
echo "================================================================="
log "Starting installation process for Ubuntu 24 EC2 instance"
log "Target: GPU instance with CUDA 12.1 support"

# Check Ubuntu version
if ! lsb_release -d | grep -q "Ubuntu 24"; then
    log_warning "This script is designed for Ubuntu 24. Current version:"
    lsb_release -d
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Installation cancelled."
        exit 1
    fi
fi

# Step 1: System packages update and installation
log "Step 1: Updating system packages and installing dependencies"
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential build tools and system dependencies
log "Installing build essentials and system dependencies..."
sudo apt-get install -y \
    build-essential \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1-dev \
    libfreetype6-dev \
    pkg-config \
    software-properties-common

log_success "System dependencies installed"

# Step 2: Verify Python 3.12
log "Step 2: Verifying Python 3.12 installation"
if ! python3.12 --version > /dev/null 2>&1; then
    log_error "Python 3.12 is not available. Installing..."
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi

PYTHON_VERSION=$(python3.12 --version)
log_success "Python version: $PYTHON_VERSION"

# Step 3: Verify CUDA availability
log "Step 3: Verifying CUDA installation"
if command -v nvcc > /dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_success "CUDA version: $CUDA_VERSION"
    
    # Check if nvidia-smi is available
    if command -v nvidia-smi > /dev/null 2>&1; then
        log "GPU information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        log_warning "nvidia-smi not available, but CUDA compiler found"
    fi
else
    log_error "CUDA not found. This script requires CUDA 12.1 for GPU training."
    log_error "Please ensure you're using a GPU-enabled EC2 instance with CUDA drivers."
    exit 1
fi

# Step 4: Create and activate virtual environment
log "Step 4: Creating Python virtual environment"
VENV_DIR="./venv"

if [ -d "$VENV_DIR" ]; then
    log_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi

python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    log_success "Virtual environment activated: $VIRTUAL_ENV"
else
    log_error "Failed to activate virtual environment"
    exit 1
fi

# Step 5: Upgrade pip and essential tools in venv
log "Step 5: Upgrading pip and build tools in virtual environment"
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel build

PIP_VERSION=$(pip --version)
log_success "Pip version: $PIP_VERSION"

# Step 6: Install PyTorch with CUDA 12.1 support
log "Step 6: Installing PyTorch with CUDA 12.1 support"
log "This may take several minutes..."

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA support
log "Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA not available in PyTorch!')
    exit(1)
"

if [ $? -ne 0 ]; then
    log_error "PyTorch CUDA verification failed!"
    exit 1
fi

log_success "PyTorch with CUDA support installed successfully"

# Step 7: Install PyTorch Geometric ecosystem
log "Step 7: Installing PyTorch Geometric ecosystem"
log "Installing torch_cluster, torch_geometric, torch_scatter, torch_sparse..."

pip install torch_cluster==1.6.3 torch_geometric==2.5.3 torch_scatter==2.1.2 torch_sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Verify PyTorch Geometric installation
log "Verifying PyTorch Geometric installation..."
python -c "
try:
    import torch_geometric
    import torch_cluster
    import torch_scatter
    import torch_sparse
    print(f'PyTorch Geometric version: {torch_geometric.__version__}')
    print('All PyTorch Geometric components imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    log_error "PyTorch Geometric installation verification failed!"
    exit 1
fi

log_success "PyTorch Geometric ecosystem installed successfully"

# Step 8: Install project requirements
log "Step 8: Installing project requirements from requirements.txt"

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found in current directory"
    exit 1
fi

pip install -r requirements.txt

# Install additional GPU monitoring tool
log "Installing additional GPU monitoring tools..."
pip install GPUtil

log_success "Project requirements installed successfully"

# Step 9: Install the neural mesh simplification package
log "Step 9: Installing neural-mesh-simplification package in development mode"

# First, uninstall any existing installation to avoid conflicts
pip uninstall -y neural-mesh-simplification 2>/dev/null || true

# Install in development mode
pip install -e .

log_success "Neural mesh simplification package installed in development mode"

# Step 10: Create necessary directories
log "Step 10: Creating necessary directory structure"

mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs

log_success "Directory structure created"

# Step 11: Verification tests
log "Step 11: Running comprehensive verification tests"

# Test all major imports
log "Testing package imports..."
python -c "
import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f'✓ {description}')
        return True
    except ImportError as e:
        print(f'✗ {description}: {e}')
        return False

success = True
print('Testing core dependencies:')
success &= test_import('torch', 'PyTorch')
success &= test_import('torchvision', 'TorchVision')
success &= test_import('torch_geometric', 'PyTorch Geometric')
success &= test_import('torch_cluster', 'PyTorch Cluster')
success &= test_import('torch_scatter', 'PyTorch Scatter')
success &= test_import('torch_sparse', 'PyTorch Sparse')

print('\nTesting scientific computing libraries:')
success &= test_import('numpy', 'NumPy')
success &= test_import('scipy', 'SciPy')
success &= test_import('sklearn', 'Scikit-learn')
success &= test_import('pandas', 'Pandas')
success &= test_import('trimesh', 'Trimesh')

print('\nTesting project modules:')
success &= test_import('neural_mesh_simplification', 'Neural Mesh Simplification')
success &= test_import('neural_mesh_simplification.models', 'NMS Models')
success &= test_import('neural_mesh_simplification.data', 'NMS Data')
success &= test_import('neural_mesh_simplification.losses', 'NMS Losses')
success &= test_import('neural_mesh_simplification.metrics', 'NMS Metrics')

print('\nTesting utility libraries:')
success &= test_import('yaml', 'PyYAML')
success &= test_import('tqdm', 'TQDM')
success &= test_import('psutil', 'PSUtil')
success &= test_import('GPUtil', 'GPUtil')

if not success:
    print('\n❌ Some imports failed!')
    sys.exit(1)
else:
    print('\n✅ All imports successful!')
"

if [ $? -ne 0 ]; then
    log_error "Import verification failed!"
    exit 1
fi

# Test CUDA functionality
log "Testing CUDA functionality..."
python -c "
import torch
import torch_geometric

print('CUDA Test Results:')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Compute capability: {props.major}.{props.minor}')
        print(f'  Total memory: {props.total_memory / 1024**3:.2f} GB')
    
    # Test tensor creation on GPU
    try:
        x = torch.rand(100, 100).cuda()
        y = torch.rand(100, 100).cuda()
        z = torch.mm(x, y)
        print('✅ GPU tensor operations successful')
    except Exception as e:
        print(f'❌ GPU tensor operations failed: {e}')
        exit(1)
else:
    print('❌ CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    log_error "CUDA functionality test failed!"
    exit 1
fi

# Test project scripts
log "Testing project scripts availability..."
for script in scripts/train.py scripts/evaluate.py scripts/infer.py scripts/preprocess_data.py; do
    if [ -f "$script" ]; then
        log_success "Found: $script"
    else
        log_error "Missing: $script"
        exit 1
    fi
done

# Test configuration files
log "Testing configuration files..."
for config in configs/default.yaml; do
    if [ -f "$config" ]; then
        log_success "Found: $config"
    else
        log_error "Missing: $config"
        exit 1
    fi
done

log_success "All verification tests passed!"

# Step 12: Display installation summary
log "Step 12: Installation Summary"

echo "================================================================="
echo "🎉 INSTALLATION COMPLETED SUCCESSFULLY! 🎉"
echo "================================================================="

echo ""
echo "📋 Installation Summary:"
echo "  ✅ System dependencies installed"
echo "  ✅ Python 3.12 virtual environment created"
echo "  ✅ PyTorch 2.4.0 with CUDA 12.1 support"
echo "  ✅ PyTorch Geometric ecosystem"
echo "  ✅ All project requirements"
echo "  ✅ Neural mesh simplification package (development mode)"
echo "  ✅ Directory structure created"
echo "  ✅ All verification tests passed"

echo ""
echo "📁 Created Directories:"
echo "  - data/raw/          (for input mesh files)"
echo "  - data/processed/    (for preprocessed training data)"
echo "  - checkpoints/       (for model checkpoints)"
echo "  - logs/             (for training logs)"

echo ""
echo "🖥️  GPU Information:"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    echo "  nvidia-smi not available"
fi

echo ""
echo "🐍 Python Environment:"
echo "  Virtual environment: $(pwd)/venv"
echo "  Python version: $(python --version)"
echo "  Pip version: $(pip --version)"

echo ""
echo "🚀 Next Steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Download training data (optional):"
echo "     python scripts/download_test_meshes.py"
echo ""
echo "  3. Preprocess your mesh data:"
echo "     python scripts/preprocess_data.py"
echo ""
echo "  4. Start training:"
echo "     python scripts/train.py --config configs/default.yaml"
echo ""
echo "  5. Or use the Jupyter notebook:"
echo "     jupyter lab train.ipynb"

echo ""
echo "⚠️  Important Notes:"
echo "  - Always activate the virtual environment before running scripts"
echo "  - The virtual environment path: $(pwd)/venv"
echo "  - CUDA ${CUDA_VERSION} is available and working"
echo "  - All scripts are ready to use"

echo ""
echo "🔧 Troubleshooting:"
echo "  - If you encounter CUDA issues, verify nvidia-smi works"
echo "  - If imports fail, ensure virtual environment is activated"
echo "  - Check logs/ directory for detailed training logs"
echo "  - Run 'python scripts/check_device.py' to verify GPU setup"

echo ""
echo "================================================================="

# Create activation helper script
log "Creating activation helper script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for neural mesh simplification environment
source venv/bin/activate
echo "🐍 Neural Mesh Simplification environment activated!"
echo "Python: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Ready to run training scripts!"
EOF

chmod +x activate_env.sh
log_success "Created activate_env.sh helper script"

echo ""
echo "💡 Quick start: Run './activate_env.sh' to activate the environment"
echo ""

log_success "Installation script completed successfully!"

# Deactivate virtual environment
deactivate 2>/dev/null || true

echo "🏁 Installation finished. Virtual environment deactivated."
echo "Run 'source venv/bin/activate' when ready to use the environment."
