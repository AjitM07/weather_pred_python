import pandas
import numpy
import matplotlib.pyplot as plt


# Simulating 30 days of temperatures
numpy.random.seed(42)
temps = numpy.random.randint(25, 35, size=30)

# Create a DataFrame for day and temperature
df = pandas.DataFrame({'Day': pandas.date_range(start='2025-06-21', periods=30),
                   'Temperature': temps})
print(df.head())


df['7_day_avg'] = df['Temperature'].rolling(window=7).mean()

# Predict tomorrow's temperature using last 7 days
last_7 = df['Temperature'][-7:]
predicted_temp = numpy.mean(last_7)
print(f"Predicted temperature for tomorrow: {predicted_temp:.2f}°C")


#visualize output by graph
plt.figure(figsize=(10,5))
plt.plot(df['Day'], df['Temperature'], label='Actual Temp', marker='o')
plt.plot(df['Day'], df['7_day_avg'], label='7-Day Avg', linestyle='--')
plt.axhline(y=predicted_temp, color='red', linestyle=':', label="Predicted Tomorrow")
plt.title('Temperature Trend & Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
