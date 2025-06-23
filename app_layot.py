import tkinter as tk
from tkinter import ttk

import pandas as pd

import Stocks
from ModelPrediction import get_train_test_data, evaluate_classification_models
from when_started import get_stocks_name


# def on_submit():
#     typed_text = input_field.get()
#     selected_option = dropdown.get()
#     output_label.config(text=f"You typed: {typed_text}\nYou selected: {selected_option}")
def on_submit():
    # Get text from dropdown
    selected_text = dropdown.get()

    try:
        X_train, X_test, Y_train, Y_test = get_train_test_data()
        results = evaluate_classification_models(X_train, X_test, Y_train, Y_test)
        best_model = max(results.items(), key=lambda item: item[1]["F1-score"])
        output_label.config(text=f"Model: {best_model[0]}")
        # if Stocks.error_code_from_download==1:
        #     output_label.config(text="Please select another stock!")
        # for name, evaluate in results.items():
        #     output_label.config(text="Model: {name}")
        #     output_label.config(text="evaluate: {evaluate}")

    except ValueError:
        output_label.config(text="Please select another stock!")

# Create main window
root = tk.Tk()
root.title("Stock App")

# Input field
input_field = tk.Entry(root, width=50)
input_field.pack(pady=20)

# Dropdown
df_stocks = pd.read_json('Stocks_name.json')
options = list(df_stocks["symbol"])
dropdown = ttk.Combobox(root, values=options)
dropdown.current(0)
dropdown.pack(pady=10)

# Button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

# Output label
output_label = tk.Label(root, text="Output will appear here")
output_label.pack(pady=20)

root.mainloop()