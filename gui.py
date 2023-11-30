from tkinter import *
from tkinter import ttk
import tkinter.messagebox as tsg
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go


def readData():
	try:
		# Update the CSV file path if necessary
		df = pd.read_csv("TSLA.csv")

		# Create a new window to display the first 10 rows of the data
		new_window = Toplevel(root)
		new_window.title("First 10 Rows of Data")
		new_window.geometry("600x400")

		# Display the first 10 rows in a text widget
		text_widget = Text(new_window, wrap=WORD)
		text_widget.pack(expand=YES, fill=BOTH)

		# Insert the first 10 rows into the text widget
		text_widget.insert(END, df.head(10))

	except Exception as e:
		tsg.showerror("Error", str(e))


def performStatisticalAnalysis():
	try:
		# Update the CSV file path if necessary
		df = pd.read_csv("TSLA.csv")

		# Perform statistical analysis, for example, calculate mean and standard deviation
		mean_close = df['Close'].mean()
		std_close = df['Close'].std()
		ath = df["High"].max()
		atl = df["Low"].min()

		# Create a new window to display statistical analysis results
		new_window = Toplevel(root)
		new_window.title("Statistical Analysis Results")
		new_window.geometry("400x200")

		# Display statistical analysis results in a label
		result_label_statistical = Label(new_window,
										 text=f"Mean Close Price: {mean_close:.2f}, Std Dev Close Price: {std_close:.2f}\n All time high: {ath} All time low: {atl}",
										 font=("Helvetica", 12))
		result_label_statistical.pack(pady=10)

	except Exception as e:
		tsg.showerror("Error", str(e))


def showCandleGraph():
	try:
		# Update the CSV file path if necessary
		df = pd.read_csv("TSLA.csv")

		# Convert the "Date" column to datetime if not already in datetime format
		if not pd.api.types.is_datetime64_ns_dtype(df['Date']):
			df["Date"] = pd.to_datetime(df["Date"])

		# Create a new window to display the candlestick graph
		new_window = Toplevel(root)
		new_window.title('Candlestick Chart')
		new_window.geometry('800x600')

		# Create a figure and axis for the candlestick chart
		fig, ax = plt.subplots(figsize=(8, 6))

		# Plot the candlestick chart
		candlestick = ax.plot(df["Date"], df["Close"], label='Candlestick Chart', color='blue')

		# Add labels and title
		ax.set_xlabel('Date')
		ax.set_ylabel('Stock Price')
		ax.set_title('Candlestick Chart')

		# Add legend
		ax.legend()

		# Embed the plot in the Tkinter window
		canvas = FigureCanvasTkAgg(fig, master=new_window)
		canvas.draw()
		canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

	except Exception as e:
		tsg.showerror("Error", str(e))


def getOption(event):
	selectedOption = initial_option.get()

	if selectedOption == "Read Data":
		readData()
	elif selectedOption == "Candle Graph":
		showCandleGraph()
	elif selectedOption == "Statistical Analysis":
		performStatisticalAnalysis()


root = Tk()
root.title("Stock Growth Prediction")
root.geometry("800x800")
root.configure(bg="lightblue")

frame = Frame(root, bg='white')
frame.pack(pady=20, padx=20)

label = Label(frame, text="Choose the Option", font=("Helvetica", 12), bg='white')
label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

initial_option = StringVar()

firstOptions = ttk.Combobox(root, textvariable=initial_option, state="readonly", font=("Helvetica", 12))
firstOptions['values'] = ["Read Data", "Candle Graph", "Statistical Analysis"]
firstOptions.pack(pady=10, padx=10)

analyze_button = Button(frame, text="Go", command=lambda: getOption(None), font=("Helvetica", 12), bg="lightblue")
analyze_button.grid(row=2, column=0, pady=30)

result_label = Label(frame, text="", font=("Helvetica", 12), bg='white')
result_label.grid(row=3, column=0, pady=10, padx=10, sticky="w")

root.mainloop()
