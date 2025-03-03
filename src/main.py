import tkinter as tk
from gui import PatientGUI

def main():
    root = tk.Tk()
    root.title("Patient Probability Viewer")
    app = PatientGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()