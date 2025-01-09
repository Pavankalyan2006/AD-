def validate_password(password):
    if len(password) >= 8:
        if any(char.isdigit() for char in password):
            if any(char.isalpha() for char in password):
                if any(char in "!@#$%^&*()_+-=[]{}|;:,.<>?/" for char in password):
                    print("Password is valid.")
                else:
                    print("Password is invalid. It must contain at least one special character.")
            else:
                print("Password is invalid. It must contain at least one letter.")
        else:
            print("Password is invalid. It must contain at least one digit.")
    else:
        print("Password is invalid. It must be at least 8 characters long.")

password = input("Enter your password: ")
validate_password(password)

def check_wine_quality(age):
    if age > 10:
        print("The wine is excellent.")
    elif age > 5:
        print("The wine is good.")
    else:
        print("The wine is average.")

age = int(input("Enter the age of the wine: "))
check_wine_quality(age)

def check_major_or_minor(age):
    if age >= 18:
        print("The person is a major.")
    else:
        print("The person is a minor.")

person_age = int(input("Enter the age of the person: "))
check_major_or_minor(person_age)

def check_purchase_availability(amount):
    if amount > 1000:
        print("Purchase is available.")
    else:
        print("Purchase is not available.")

purchase_amount = float(input("Enter the purchase amount: "))
check_purchase_availability(purchase_amount)
