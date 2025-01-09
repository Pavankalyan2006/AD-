import pandas as pd

data = {
    "Name": ["John", "Alice", "Bob", "Diana"],
    "Age": [28, 34, 23, 29],
    "Department": ["HR", "IT", "Marketing", "Finance"],
    "Salary": [45000, 60000, 35000, 50000]
}
df = pd.DataFrame(data)
df
first= df.head(2)
print(first)
df['Bonus'] = df['Salary'] * 0.1
print(df)
average = df['Salary'].mean()
print(average)
employees= df[df['Age'] > 25]
print(employees)
