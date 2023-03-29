# Parallel_Sorting_CUDA
In this project two sorting algorithms, Bitonic and Merge sort, have been re-implemented in their parallel versions using the CUDA exten- sion for the C programming language.

## Run commands (Windows)
Compile and execute bitonic sort:
'''
nvcc bitonic.cu -o bitonic.exe
./bitonic.exe
'''

Compile and execute merge sort:
'''
nvcc merge.cu -o merge.exe
./merge.exe
'''

## Run commands (Linux)
Compile and execute bitonic sort:
'''
nvcc bitonic.cu -o bitonic.out
./bitonic.out
'''

Compile and execute merge sort:
'''
nvcc merge.cu -o merge.out
./merge.out
'''

## Time & Throughput comparison (CUDA vs CPU)
<img src="resources/time.png" width="50%"> <img src="resources/throughput.png" width="44.5%"/> 


