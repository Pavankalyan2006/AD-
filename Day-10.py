import pandas as pd
data = pd.read_csv("C:\\Users\\pavan\\Downloads\\Day_10_banking_data.csv")
filtered = data[data["Transaction_Amount"] > 2000]
loan = data[(data["Transaction_Type"] == "Loan Payment") & (data["Account_Balance"] > 5000)]
uptown = data[data["Branch"] == "Uptown"]
print(filtered)
print(loan)
print(uptown)
data["Transaction_Fee"] = data["Transaction_Amount"] * 0.02
data["Balance_Status"] = data["Account_Balance"].apply(lambda x: "High Balance" if x > 5000 else "Low Balance")
print(data)
