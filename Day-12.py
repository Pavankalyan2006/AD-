import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\pavan\\Downloads\\Day_12_banking_data.csv")
transaction = data.groupby("Account_Type")["Transaction_Amount"].sum()
transaction.plot(kind="bar", title="Total Transaction Amount per Account Type")
plt.xlabel("Account Type")
plt.ylabel("Total Transaction Amount")
plt.show()
branch_transaction_count = data["Branch"].value_counts()
branch_transaction_count.plot(kind="pie", autopct="%1.1f%%", title="Percentage of Transactions per Branch")
plt.ylabel("") 
plt.show()
