#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 12*1024
#define MAX_LENGTH 512*1024*1024
#define MAX_LENGTH_SEQ 1*1024*1024
#define REPETITIONS 5


// UTILS
void randomArray(int *a, int n){
/*
    Randomly initialize the elements of an array given its size.
*/
    for(int i=0; i < n; i++){
        a[i] = rand();
    }
}


void copyArray(int *a, int *b, int n){
/*
    Copy elements from one array to another.
*/
    for(int i=0; i < n; i++){
        b[i] = a[i];
    }
}


__global__ void copyArrayDevice(int *dst, int *src, int n, int blocks){
/*
    Copy elements from one array in the DEVICE memory into another one in the DEVICE memory.
*/
    int threads = blocks * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=idx; i < n; i+=threads){
        dst[i] = src[i];
    }
}


bool compareArrays(int *a, int *b, int n){
/*
    Compare the elements of two arrays.
*/
    for(int i=0; i < n; i++){
        if(a[i] != b[i]){
            return false;
        }
    }
    return true;
}


bool everIncreasing(int *a, int n){
/*
    Check wheter the array is monotonically increasing.
*/
    for(int i=1; i < n; i++){
        if(a[i] < a[i-1]){
            return false;
        }
    }
    return true;
}


void printResults(double *time, double *thr, int n){
    for(int r=0; r < n; r++){
        printf("%.4f,    ", time[r]);
    }
    printf("(s)\n");
    for(int r=0; r < n; r++){
        printf("%.0f,    ", thr[r]);
    }
    printf("(el/s)\n\n");
}


__host__ __device__ int binarySearchCount(int val, int *a, int first, int last, int ord){
/*
    Count the number of elements that are smaller than 'val' (also equal if 'ord' is 1).
*/
    int left = first;
    int right = last;
    int mid;
    while(left < right){
        mid = (right + left) / 2;
        
        if(a[mid] == val && ord == 1){
            while(mid + 1 < last && a[mid+1] == val){
                mid++;
            }
            return mid + 1 - first;
        }
            
        if(a[mid] == val && ord == 0){
            while(mid - 1 >= first && a[mid-1] == val){
                mid--;
            }
            return mid - first;       
        }

        if(a[mid] > val){
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    while(mid >= first && a[mid] > val){
        mid--;
    }
    return mid + 1 - first;
}


// SEQUENTIAL VERSION
void mergeSortSeq(int *a, int n){
    int *aux = (int *)malloc(n * sizeof(int));
    int ord, before, first, last;

    // Sub-arrays
    for(int size=1; size < n; size*=2){

        // Element sorting
        for(int i=0; i < n; i++){
            ord = (i/size) % 2;                             // Indicates if first or second array of the pair
            before = i%size + (i/(2*size)) * 2 * size;      // N° of elements before the current pair of sub-arrays
            first = (i/size + 1 - 2*ord) * size;            // Index of the first element of the sub-array used for binary search
            last = min(first + size, n);                    // Index of the last element of the sub-array used for binary search

            // Copy the elements in the right position
            if(last > first){
                int idx = binarySearchCount(a[i], a, first, last, ord);
                aux[idx + before] = a[i];
            } else {
                aux[before] = a[i];
            }
        }
        // Move the values back from the auxiliary array into the original array
        copyArray(aux, a, n);
    }
}


// PARALLEL VERSION
__global__ void mergeInMem(int *a, int n, int blocks){
/*
    Performs multiple steps of merge sort, until data does not fit the shared memory anymore.

    In these kernel all the computations are performed using the shared memory.
*/ 

    int elems_block = (n + blocks - 1) / blocks;    // Elements per block
    int off = blockIdx.x * elems_block;             // Block offset

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS/2];
    __shared__ int aux[MAX_SHARED_ELEMS/2];

    // Fill shared memory: each block loads the values in a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }

    int ord, before, first, last;
    for(int size=1; size < elems_block; size*=2){
        __syncthreads();
        for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
            ord = (i/size) % 2;                             // Indicates if first or second array of the pair
            before = i%size + (i/(2*size)) * 2 * size;      // N° of elements before the current pair of sub-arrays 
            first = (i/size + 1 - 2*ord) * size;            // Index of the first element of the sub-array used for binary search
            last = min(first + size, elems_block);          
            last = min(last, n - off);                      // Index of the last element of the sub-array used for binary search
            
            // Copy the elements in the right position
            if(last > first){
                int idx = binarySearchCount(v[i], v, first, last, ord);
                aux[idx + before] = v[i]; 
            } else {
                aux[before] = v[i];
            }
            
        }
        __syncthreads();
        // Move the values back from the auxiliary array into the original array
        for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
            v[i] = aux[i]; 
        }
    }

    // Write back the data from shared to global memory
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }
}


__global__ void mergeOffMem(int *a, int *aux, int n, int size, int blocks){
/*
    Performs one step of merge sort. 
    In particular, it computes the step in which the sub-arrays have a numer of elements equal to 'size'.

    In these kernel all the computations are performed using the global memory.
*/ 
    int elems_block = (n + blocks - 1) / blocks;    // Elements per block
    int off = blockIdx.x * elems_block;             // Block offset

    int i, ord, before, first, last; 
    for(int j=threadIdx.x; (j < elems_block) && (j + off < n); j+=blockDim.x){
        i = j + off;
        ord = (i/size) % 2;                             // Indicates if first or second array of the pair
        before = i%size + (i/(2*size)) * 2 * size;      // N° of elements before the current pair of sub-arrays
        first = (i/size + 1 - 2*ord) * size;            // Index of the first element of the sub-array used for binary search
        last = min(first + size, n);                    // Index of the last element of the sub-array used for binary search
        
        // Copy the elements in the right position
        if(last > first){
            int idx = binarySearchCount(a[i], a, first, last, ord);
            aux[idx + before] = a[i];
        } else {
            aux[before] = a[i];
        }
    }
}


void mergeSortPar(int *a, int n){
/*
    Merge sort is a sorting algorithm with log_2(n) merging steps.
    
    Advantages:
        - The sequential version is already very efficient.
        - Parallel time complexity: θ( (n/p) * log(n) )

    Disadvantages:
        - The original merge function is not good for parallel implementations (here binary search is used).
        - Not in-place (requires an auxiliary array -> doubles the memory requirements)
*/

    int *d_a, *d_aux;
	const size_t size = n * sizeof(int);
    
    cudaMalloc((void **)&d_a, size);      // Allocate space in DEVICE memory
    cudaMalloc((void **)&d_aux, size);    // Allocate space in DEVICE memory

    // Copy the auxiliary array into the DEVICE 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    // Use the number of blocks that maximize the utilization of shared memory
    int blocks = (n + double(MAX_SHARED_ELEMS/2) - 1) / double(MAX_SHARED_ELEMS/2);
    int threads = 1024;
    int elems_block = (n + blocks - 1) / blocks;
    
    // Initial merging steps using the SHARED memory
    mergeInMem<<<blocks, threads>>>(d_a, n, blocks);
    cudaDeviceSynchronize(); // Wait for all the blocks
    for(int size=elems_block; size < n; size*=2){
        
        // Continue merging using the GLOBAL memory
        mergeOffMem<<<blocks, threads>>>(d_a, d_aux, n, size, blocks);
        cudaDeviceSynchronize(); // Wait for all the blocks

        // Copy data from the auxiliary array to the original one
        copyArrayDevice<<<blocks, threads>>>(d_a, d_aux, n, blocks);
        cudaDeviceSynchronize(); // Wait for all the blocks
    }
   
    // Copy the results from the array in the DEVICE to the one in the HOST memory
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    // Free DEVICE memory
    cudaFree(d_a); cudaFree(d_aux);
}


// MAIN
int main(){

    int *a, *b;
    clock_t start, stop;
    double timeS[REPETITIONS], timeP[REPETITIONS];
    double thrS[REPETITIONS], thrP[REPETITIONS];

    for(int length=1024; length <= MAX_LENGTH; length*=2){
        printf("\n\n############################################");
        printf("############################################\n\n");
        printf("N = %d\n\n", length);
        for(int r=0; r < REPETITIONS; r++){
            
            // Allocate memory for the array
            a = (int *)malloc(length * sizeof(int));
            b = (int *)malloc(length * sizeof(int));

            // Initialize random array
            randomArray(a, length);
            copyArray(a, b, length);

            if(length <= MAX_LENGTH_SEQ){
                // SEQUENTIAL algorithm
                start = clock();
                mergeSortSeq(a, length);
                stop = clock();
            }

            timeS[r] = double(stop-start+1) / double(CLOCKS_PER_SEC);
            thrS[r] = double(length * (double(CLOCKS_PER_SEC) / double(stop-start+1)));

            // PARALLEL algorithm
            start = clock();
            mergeSortPar(b, length);
            stop = clock();

            timeP[r] = double(stop-start+1) / double(CLOCKS_PER_SEC);
            thrP[r] = double(length * (double(CLOCKS_PER_SEC) / double(stop-start+1)));

            // Check correctness
            if((length <= MAX_LENGTH_SEQ && compareArrays(a, b, length) != true) ||
                    (length > MAX_LENGTH_SEQ && everIncreasing(b, length) != true)){
                printf("\nERROR!!\n");
            }

            free(a); free(b);
        }
        if(length <= MAX_LENGTH_SEQ){
            printf("SEQ\n");
            printResults(timeS, thrS, REPETITIONS);
        }
        printf("PAR\n");
        printResults(timeP, thrP, REPETITIONS);
    }
}
