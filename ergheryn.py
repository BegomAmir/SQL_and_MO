score = int(input())
def grade(score):
    if score < 60:
        print('Неудовлетворительно')
    elif score <= 74:
        print('Удовлетворительно')
    elif score <= 90:
        print('Хорошо')
    elif score <= 100:
        print('Отлично')
    return grade
print(grade())