# GADA: Reconfiguration of Non-Crossing Spanning Trees

Various code to explore the bounds of the flip distance between non-crossing trees in convex position.

We specifically aim to find a pair of tree with a new best-know confliction value $\gamma$.

## Setup

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository:** TODO: Update this after the repo is done
   ```bash
   git clone <repository-url>
   cd reconfiguration-non-crossing-spanning-trees
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv Scripts/venv
   source Scripts/venv/bin/activate  # On Windows: Scripts\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   cd Scripts
   make install-deps  # If requirements.txt exists
   # OR manually:
   # pip install -r requirements.txt
   ```

4. **Install in development mode (optional):**
   ```bash
   make install-dev
   ```

## Usage

### Using the Makefile (Recommended)

Navigate to the Scripts directory and use the provided Makefile:

```bash
cd Scripts

# Run the main application
make run

# Run the demo
make demo

# Show available commands
make help

# Debug project setup
make debug
```

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make run` | Run Scripts.main as module (recommended) |
| `make run-direct` | Run main.py directly with PYTHONPATH |
| `make run-local` | Run from Scripts directory |
| `make demo` | Run the demonstration script |
| `make install-dev` | Install package in development mode |
| `make install-deps` | Install dependencies from requirements.txt |
| `make check-venv` | Verify virtual environment exists |
| `make activate` | Show command to activate virtual environment |
| `make clean` | Clean Python cache files |
| `make debug` | Show project info and structure |
| `make help` | Show all available commands |

### Manual Execution

If you prefer not to use the Makefile:

```bash
# From project root
cd reconfiguration-non-crossing-spanning-trees
PYTHONPATH=. Scripts/venv/bin/python -m Scripts.main

# Or activate virtual environment first
source Scripts/venv/bin/activate
python -m Scripts.main
```

## Development

### Import Structure

The project uses Python's module system with proper package structure. Imports follow this pattern:

```python
# From within Scripts package
from .analysis import GammaAnalyzer
from .core import SomeClass
```

### Adding New Features

1. Create new modules in the `Scripts/` directory
2. Add `__init__.py` files for new subdirectories
3. Use relative imports within the package
4. Update the Makefile if new entry points are needed

## Troubleshooting

### Import Errors

If you encounter import errors like `ImportError: attempted relative import with no known parent package`:

1. **Use the Makefile**: `make run` handles Python path correctly
2. **Check virtual environment**: `make check-venv`
3. **Use module execution**: `python -m Scripts.main` instead of `python main.py`
4. **Verify project structure**: `make debug`

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf Scripts/venv
python -m venv Scripts/venv
source Scripts/venv/bin/activate
pip install -r requirements.txt  # if exists
```

### LSP/IDE Configuration

For proper IDE support with imports, consider:

1. **Configure Python path** in your IDE to include project root
2. **Install in development mode**: `make install-dev`
3. **Use absolute imports** if relative imports cause LSP warnings

## Contributing

1. Follow the existing code structure and import patterns
2. Add new functionality to appropriate modules
3. Update this README for significant changes
4. Test using the provided Makefile commands

## License

TODO: [Add license information here]

## Contact

TODO: [Add contact information or research group details here]
