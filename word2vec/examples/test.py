
filename = 'Training.txt'

with open(filename, 'r') as file_to_read:
    while True:
        text_to_annotate = file_to_read.readline()
        break

filename = 'New_Training.txt'

with open(filename, 'r') as file_to_read:
    while True:
        text_to_annotate2 = file_to_read.readline()
        break
text_to_annotate = text_to_annotate.lower()

str = ''
for i in range(len(text_to_annotate)):
    if text_to_annotate[i] == text_to_annotate2[i]:
        str += text_to_annotate[i]
    else:
        print(str)
        break
#print(text_to_annotate == text_to_annotate2)
print("\n")
print(text_to_annotate)
print(text_to_annotate2)
