from .examples import *

if __name__ == "__main__":
    # Get all callables from current module whose names end with '_demo'
    demo_functions = [
        (name, obj)
        for name, obj in globals().items()
        if callable(obj) and name.endswith("_demo")
    ]

    # Sort for consistent order (optional)
    demo_functions.sort()

    # Display the options
    print("Choose a demo to run:")
    for i, (name, _) in enumerate(demo_functions):
        print(f"{i}: {name}")

    # Get user input
    choice = input("Enter the number of the demo to run: ")

    try:
        index = int(choice)
        _, selected_function = demo_functions[index]
        print(f"Running: {demo_functions[index][0]}\n")
        selected_function()
    except (ValueError, IndexError):
        print("Invalid selection.")
