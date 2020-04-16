text = open("lorem.input.txt").read().strip()
with open("lorem.txt", "w") as f:
    f.write(text)
