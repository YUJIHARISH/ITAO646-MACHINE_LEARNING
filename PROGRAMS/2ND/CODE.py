import csv

def generalize_S(specific, example):
    """Generalize the specific boundary."""
    return ['?' if specific[i] != example[i] else specific[i] for i in range(len(specific))]

def specialize_G(general, example, domains):
    """Specialize the general boundary."""
    result = []
    for hypothesis in general:
        for i in range(len(hypothesis)):
            if hypothesis[i] == '?':
                for value in domains[i]:
                    if value != example[i]:
                        new_hypothesis = hypothesis[:]
                        new_hypothesis[i] = value
                        result.append(new_hypothesis)
            elif hypothesis[i] != example[i]:
                new_hypothesis = hypothesis[:]
                new_hypothesis[i] = '?'
                result.append(new_hypothesis)
    return result

def candidate_elimination(file_path):

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

   
    attributes = data[0][:-1]  
    examples = data[1:]

    S = ['0' for _ in range(len(attributes))]
    G = [['?' for _ in range(len(attributes))]]


    domains = [set() for _ in range(len(attributes))]
    for example in examples:
        for i, value in enumerate(example[:-1]):
            domains[i].add(value)

    for example in examples:
        instance = example[:-1]
        label = example[-1]

        if label == 'Yes':
 
            G = [g for g in G if all(g[i] == '?' or g[i] == instance[i] for i in range(len(instance)))]
            S = generalize_S(S, instance)
        else:
   
            S = S
            G = specialize_G(G, instance, domains)
            G = [g for g in G if any(s != '0' and g[i] != '?' and s != g[i] for i, s in enumerate(S))]

    return S, G


file_path = r'C:\Users\Administrator\Documents\ML Datasets\2.candidate_elimination.csv'
S, G = candidate_elimination(file_path)

print("Final Version Space:")
print("S (Specific Boundary):", S)
print("G (General Boundary):", G)
