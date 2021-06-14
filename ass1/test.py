import re
string = "11 john's rest\/is here! Hello, my friend"
rtn = re.findall(r'\w+', string)
rtn = " ".join(filter(lambda x: len(x) > 1, rtn))
print(rtn)