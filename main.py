import argparse
import torch

from model import GalaxyClassifierS4D, ModelInterface, GalaxyExplorerGUI
from model.functions import load_data

def parse_arguments():
    """
    Parse command line arguments for the Galaxy Explorer GUI.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - python : bool, whether to use Python model implementation
        - model_path : str, path to the trained model file
        - riscv : bool, whether to use RISC-V implementation
        - colored : bool, whether to use colored images
    """
    parser = argparse.ArgumentParser(
        description="Interactive Galaxy Classification Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --python -m galaxy_model.pth
  %(prog)s -p -m galaxy_model.pth --colored
  %(prog)s --riscv
        """
    )
    
    # Model implementation group
    impl_group = parser.add_mutually_exclusive_group(required=True)
    impl_group.add_argument(
        '--python', '-p',
        action='store_true',
        help='Use Python model implementation'
    )
    impl_group.add_argument(
        '--riscv',
        action='store_true',
        help='Use RISC-V model implementation'
    )
    
    # Model path (required for Python implementation)
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default='galaxy_s4_model.pth',
        help='Path to trained model file (default: galaxy_s4_model.pth)'
    )
    
    # Image options
    parser.add_argument(
        '--colored', '-c',
        action='store_true',
        help='Use colored (RGB) images instead of grayscale (default: grayscale)'
    )
    
    # Data directory
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Root directory for dataset (default: ./data)'
    )
    
    args = parser.parse_args()
    
    # Validate that model path is provided for Python implementation
    if args.python and not args.model_path:
        parser.error("--model-path/-m is required when using --python/-p")
    
    return args


def main():
    """
    Main entry point for the Galaxy Explorer GUI.
    
    Parses command line arguments and dispatches to appropriate handler.
    """
    args = parse_arguments()
    
    print("=" * 60)
    print("S4 Galaxy Classification Explorer")
    print("=" * 60)
    print(f"Implementation: {'Python' if args.python else 'RISC-V'}")
    print(f"Image mode: {'Colored' if args.colored else 'Grayscale'}")
    print("=" * 60)
    
    # Load test data (same for all implementations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading test dataset from {args.data_dir}...")
    
    X_test, y_test_onehot, y_test = load_data(
        root=args.data_dir,
        download=True,
        train=False,
        colored=args.colored
    )
    num_classes = y_test_onehot.shape[1]
    
    print(f"Loaded {len(X_test)} samples with {num_classes} classes")
    
    # Initialize model interface (handles Python vs RISC-V internally)
    implementation = 'python' if args.python else 'riscv'
    model = ModelInterface(
        implementation=implementation,
        model_path=args.model_path,
        num_classes=num_classes,
        colored=args.colored,
        device=device
    )
    
    # Launch GUI (same for all implementations)
    print("\nLaunching GUI (LEFT/RIGHT: navigate | R: random | M: magma | Q: quit)")
    explorer = GalaxyExplorerGUI(
        model=model,
        x_val=X_test,
        y_val=y_test_onehot,
        device=device,
    )
    explorer.run()


if __name__ == "__main__":
    main()