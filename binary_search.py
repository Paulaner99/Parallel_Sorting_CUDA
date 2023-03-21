import numpy as np
np.random.seed(0)

a = np.sort(np.random.randint(0, 10, 4))
b = np.sort(np.random.randint(0, 10, 6))
print(a, b)

def binary_search(key, v, first, last, ord):
    left = first
    right = last
    while left < right:
        mid = int((left + right) / 2)

        if v[mid] == key and ord == 1:
            while mid + 1 < last and v[mid+1] == key:
                mid += 1
            return mid + 1
        
        if v[mid] == key and ord == 0:
            while mid - 1 >= 0 and v[mid-1] == key:
                mid -= 1
            return mid
          
        if v[mid] > key:
            right = mid
        else:
            left = mid + 1
    
    while mid >= 0 and v[mid] > key:
        mid -= 1
    return mid + 1


    

for i in range(len(a)):
    pos = i+binary_search(a[i], b, 0, len(b), 0)
    print(pos)

for i in range(len(b)):
    pos = i+binary_search(b[i], a, 0, len(a), 1)
    print(pos)