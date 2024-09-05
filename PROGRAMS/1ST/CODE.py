import csv

def find_s(file_path):
  
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    attributes = data[0][:-1] 
    examples = data[1:]

   
    hypothesis = None
    for example in examples:
        if example[-1] == 'Yes':  
            hypothesis = example[:-1]
            break

   
    if hypothesis is None:
        return None

    for example in examples:
        if example[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if example[i] != hypothesis[i]:
                    hypothesis[i] = '?'

    return attributes, hypothesis


file_path = r'C:\Users\Administrator\Documents\ML Datasets\1.traning.csv'
attributes, result = find_s(file_path) 

if result:
    print("Most specific hypothesis:")
    for attr, value in zip(attributes, result):
        print(f"{attr}: {value}")
else:
    print("No hypothesis found (no positive examples in the dataset).")
