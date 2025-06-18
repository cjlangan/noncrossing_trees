# GADA: Reconfiguration of Non-Crossing Spanning Trees

Various code to explore the bounds of the flip distance between non-crossing trees in convex position.

We specifically aim to find a pair of trees with a new best-know confliction value $\gamma$ as defined in "Flipping Non-Crossing Spanning Trees" by Bjerkevik et. al.

## Setup

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

```bash
git clone https://github.com/cjlangan/noncrossing_trees
cd noncrossing_trees/ && make
```

- This will automatically create a virtual environment, install requirements, and run the program.

## Usage

### Using the Makefile (Recommended)

Navigate to the noncrossing_trees directory and use the provided Makefile:

```bash
cd noncrossing_trees

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
| `make run` | Run noncrossing_trees.main as module (recommended) |
| `make run-direct` | Run main.py directly with PYTHONPATH |
| `make run-local` | Run from noncrossing_trees directory |
| `make demo` | Run the demonstration script |
| `make install-dev` | Install package in development mode |
| `make install-deps` | Install dependencies from requirements.txt |
| `make check-venv` | Verify virtual environment exists, else creates one |
| `make activate` | Show command to activate virtual environment |
| `make clean` | Clean Python cache files |
| `make debug` | Show project info and structure |
| `make help` | Show all available commands |


## Manually Running

If you prefer to not use the Makefile, you can install and run without it:

```bash
cd noncrossing_trees/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd .. && python -m noncrossing_trees.main
```

## Development

### Import Structure

The project uses Python's module system with proper package structure. Imports follow this pattern:

```python
# From within noncrossing_trees package
from .analysis import GammaAnalyzer
from .core import TreeUtils
```

### Adding New Features

1. Create new modules in the `noncrossing_trees/` directory
2. Add `__init__.py` files for new subdirectories
3. Use relative imports within the package
4. Create a demo file in the `examples/` directory that uses your feature
5. Edit `__init__.py` in the `examples/` directory to add your demo
6. Update the main.py to use your demo file

## Troubleshooting

### Import Errors

If you encounter import errors like `ImportError: attempted relative import with no known parent package`:

1. **Use the Makefile**: `make run` handles Python path correctly
2. **Check virtual environment**: `make check-venv`
3. **Use module execution**: `python -m noncrossing_trees.main` instead of `python main.py`
4. **Verify project structure**: `make debug`

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
5. Submit a detailed PR

## License

TODO: [Add license information here when public]

## Contact

First Last | Email 
---- | -----
Connor Langan | langanc@myumanitoba.ca
Atishaya | maharjaa@myumanitoba.ca
