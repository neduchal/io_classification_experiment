

data = open("results.txt").read().split("\n")

good = 0

for row in data:
    if len(row) == 0:
        continue
    items = row.split(" ")
    if int(items[0]) == int(items[1]):
        good += 1

print(good / len(data))
