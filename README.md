# My GUI Project

This project is a Python application that provides a graphical user interface (GUI) for displaying patient probabilities. It simulates patient data and allows users to view the top 10 probabilities for each patient through a set of buttons.

## Project Structure

```
my-gui-project
├── src
│   ├── main.py          # Entry point of the application
│   ├── gui.py           # GUI layout and button behavior
│   ├── model.py         # Logic for generating patient data and probabilities
│   └── data
│       └── __init__.py  # Marks the data directory as a package
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-gui-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will launch the GUI, where you can interact with the buttons representing patient IDs and view their corresponding top 10 probabilities.

## Dependencies

The project requires the following libraries:
- Tkinter or PyQt (for GUI development)
- NumPy
- Pandas
- Scikit-learn

Make sure to check `requirements.txt` for the complete list of dependencies.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.