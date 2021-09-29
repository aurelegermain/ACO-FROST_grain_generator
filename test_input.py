import numpy as np

with open('test_input.inp' , "rt") as myfile:
    output = myfile.read()

try:    
    start_string="$grain"
    end_string="$end" 
    start = output.index(start_string) + len(start_string)
    end = output.index(end_string, start)
    input_grain = output[start:end].strip().split()
    input_grain = np.reshape(input_grain,(-1,2))
except:
    input_grain = None

print(input_grain)

try:    
    start_string="$building"
    end_string="$end" 
    start = output.index(start_string) + len(start_string)
    end = output.index(end_string, start)
    input_building = output[start:end].strip().split()
    input_building = np.reshape(input_building,(-1,2))
except:
    input_building = None

print(input_building[1:,:])

list_weight = input_building[1:,1].astype(int)/np.sum(input_building[1:,1].astype(int))
print(np.random.choice(len(input_building[1:,:]),1, p=list_weight))