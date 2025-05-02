from termcolor import colored

print(colored("Hello", "red", attrs=["bold"]),"666")
print(colored("警告", color="red", attrs=["bold"]))
print(colored("提示", color="blue", on_color="on_white"))
