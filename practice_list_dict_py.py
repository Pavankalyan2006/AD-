list = {'abd': 'as', 'hf': 'er'}
print(list)

students = {
    1: {"name": "google", "age": 20},
    2: {"name": "deloite", "age": 22},
    3: {"name": "pwc", "age": 19},
    4: {"name": "aaa", "age": 21},
    5: {"name": "Eee", "age": 23},
}

for student_id, details in students.items():
    print(f"Student ID: {student_id}, Name: {details['name']}, Age: {details['age']}")

import pandas as pd
data = [20, 30, 40, 50, 2, 3, 6, 8, 10, 18, 19, 20, 24, 2, 4, 5, 6]

No_of_studied = data[4:8]
print("No.Of hours_studied", No_of_studied)
Age = data[9:13]
print("Student_age", Age)
screen_time = data[12:18]
print("screen_time", screen_time)
Data_Frame = pd.DataFrame(data)
print(Data_Frame)

No_of_studied = [2, 3, 6, 8]
Age = [10, 18, 19, 20]
screen_time = [24, 2, 4, 5]

df = pd.DataFrame({
    "No_of_studied": No_of_studied,
    "Age": Age,
    "screen_time": screen_time
})

print(df)

numbers = [1, 2, 3, 4, 5]
print(numbers)

numbers[0] = 99
numbers[2] = 'python'
numbers[4] = -46

print(numbers)

string = input("Enter elements (space-separated): ")
lst = string.split()
print('The list is:', lst)

list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
list3 = list1 + list2

print(list3)

favorite_fruits = ["Apple", "Banana", "Cherry"]
favorite_vegetables = ["Carrot", "Broccoli", "Cucumber"]

favorite_foods = favorite_fruits + favorite_vegetables

print("Favorite Fruits:", favorite_fruits)
print("Favorite Vegetables:", favorite_vegetables)
print("Favorite Foods:", favorite_foods)

thistuple1 = ("apple",)
print(type(thistuple1))

thistuple2 = ("apple")
print(type(thistuple2))

tuple1 = ("abc", 34, True, 40, "male")
print(tuple1)

my_list = [10, 20, 30, 40]
size = len(my_list)
print(size)

d = {1: 'Geeks', 2: 'For', 3: 'Geeks'}
print(d)
