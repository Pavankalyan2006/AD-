import pandas as pd
import math
screen_time = [2, 4, 6]
sleep_hours = [3, 7, 8]
stu_name = ["karthik", "vivek", "raju"]
dict1 = {
    "screen_time": screen_time,
    "sleep_hours": sleep_hours,
    "stu_name": stu_name
}
print(dict1)
df = pd.DataFrame(dict1)
print(df)

name = ["a", "b", "c"]
id = [1, 2, 3]
phone = [123, 124, 145]
dict1 = {
    "Name": name,
    "id": id,
    "phone_number": phone
}
print(dict1)
df = pd.DataFrame(dict1)
print(df)

list = [2, 1, 13, 15]
res = [i + 2 for i in list if i < 4]
print(res)
res = [i for i in list if i % 2 == 0]
print(res)

words = ["LOWER", "W", "PYTHON"]
lower = [i.lower() for i in words]
print(lower)

words = ["madam", "radar", "abc"]
result = [i for i in words if i == i[::-1]]
print("palindrome strings are ", result)

numbers = [-5, 3, -2, 8, -1]
resu = [x if x > 0 else 0 for x in numbers]
print(resu)

keys = ["name", "age", "city"]
values = ["banglore", 25, "Hyderabad"]
dictionary = {k: v for k, v in zip(keys, values)}
print(dictionary)

list1 = [1, 2, 3, 4, 5]
list2 = [3, 5, 8, 9, 1]
result = [i for i in list1 if i in list2]
print(result)

result = [x ** 2 for x in range(1, 20)]
print(result)

dict1 = {"a": 12, "abc": 123, "dfe": 45}
result = {key: value for key, value in dict1.items()}
print(result)

my_list = [x for x in range(2500, 2586)]
print(my_list)

my_list = [x for x in range(3, 301, 3)]
print(my_list)

sum_squares = sum([x ** 2 for x in range(1, 16)])
print(sum_squares)

students = {"abc": 85, "def": 65, "ccc": 95, "chaina": 70}
passing_students = [name for name, score in students.items() if score >= 75]
grades = {name: "A" if score >= 90 else "B" for name, score in students.items()}
print(passing_students)
print(grades)

products = [
    {"name": "Laptop", "price": 800},
    {"name": "Smartphone", "price": 500},
    {"name": "Tablet", "price": 300}
]
affordable_products = {product["name"]: product["price"] for product in products if product["price"] <= 500}
print(affordable_products)

list3 = [4, 6, 36]
result = tuple(math.sqrt(i) for i in list3)
print(result)

numbers = [1, 4, 9, 16, 25]
square_roots = tuple(math.sqrt(num) for num in numbers)
print(square_roots)

square = lambda x: x ** 2
print(square(3))

def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        return "Invalid operation"

print(calculator(10, 5, "add"))
