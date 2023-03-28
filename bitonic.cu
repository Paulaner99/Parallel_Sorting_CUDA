#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 8*1024
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


void compareAndSwap(int *a, int i, int j, int dir){
/*
    Compare and swap two elements of an array, according to the direction of the ordering.
*/
    if(dir == (a[i] > a[j])){
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
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


// SEQUENTIAL VERSION
void bitonicSort(int *a, int n){
    // Steps
    for(int k=2; k <= n; k*=2){

        // Stages
        for(int j=k/2; j > 0; j/=2){
            int l, tmp;

            // Elements
            for(int i=0; i < n; i++){
                l = i ^ j;
                if(l > i){
                    // COMPARE
                    if((i & k) == 0 && (a[i] > a[l]) || (i & k) != 0 && (a[i] < a[l])){
                        // SWAP
                        tmp = a[i];
                        a[i] = a[l];
                        a[l] = tmp;
                    }
                }
            }
        }
    }
}


void bitonicSortSeq(int *a, int n){
    // Compute the nearest power of two >= n
    int m = (int)pow(2, ceil(log2(n)));
    const size_t size = m * sizeof(int);

    // Instantiate an auxiliary array with a number of elements which is a number of two 
    int *aux;
    aux = (int *)malloc(size);              // Allocate space

    // Copy values into the auxiliary array
    for(int i=0; i < m; i++){
        if(i < n){
            aux[i] = a[i];
        } else {
            aux[i] = -1;
        }
    }
    
    bitonicSort(aux, m);

    int i = 0;
    int j = 0;
    while(i < m){
        if(aux[i] != -1){
            a[j] = aux[i];
            j++;
        }
        i++;
    }
    free(aux);
}


// PARALLEL VERSION
__global__ void bitonicInit(int *a, int n, int elems_block){
/*
    Performs multiple steps of bitonic sort until dependency between data remains within blocks. 
    In particular, it computes all the boxes up to k=elems_block. This is the last block with no external dependencies.

    In these kernel all the computations are performed using the shared memory.
*/    
    
    // Block offset
    int off = blockIdx.x * elems_block;

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS];

    // Fill shared memory: each block loads the values in a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }
    __syncthreads();
    
    // Bitonic sort up to step k=elems_block
    for(int k=2; k <= elems_block; k*=2){
        for(int j=k/2; j > 0; j/=2){
            // One step of bitonic sort
            int l, i, tmp;

            // Each thread evaluates comparisons and swaps until all elements are re-ordered. 
            // (according to the pattern required at step (k, j))
            for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
                i = idx + off;
                l = i ^ j;
                if(l > i){
                    // COMPARE
                    if((i & k) == 0 && (v[idx] > v[l-off]) || (i & k) != 0 && (v[idx] < v[l-off])){
                        // SWAP
                        tmp = v[idx];
                        v[idx] = v[l-off];
                        v[l-off] = tmp;
                    }
                }
            }
            // Wait until the shared vector is completely rearranged before proceeding to the next step.
            // It is not necessary to synch at block level given that the following operations are local.
            __syncthreads();
        }
    }
    
    // Write back the data from shared to global memory
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }    
}


__global__ void bitonicInMem(int *a, int n, int k, int j, int elems_block){
/*
    Performs multiple steps of bitonic sort within the same box 'k'. 
    In particular, it computes log_2(j) steps (i.e., up to j=1, with j halved at each step).

    The value of 'j' represents the 'distance' between elements to be compared (and eventually swapped).
        -> 2*j is the number of elements on which the following steps within the 'k' box depend!

    In these kernel all the computations are performed using the shared memory.
*/    

    // Block offset
    int off = blockIdx.x * elems_block;

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS];

    // Fill shared memory: each block loads values into a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }
    __syncthreads(); // Wait for the local vector to be filled

    // Steps: j -> j/2 -> j/4 -> .. -> 1
    while(j > 0){
        // One step of bitonic sort
        int l, i, tmp;

        // Each thread evaluates comparisons and swaps until all elements are re-ordered. 
        // (according to the pattern required at step (k, j))
        for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
            i = idx + off;
            l = i ^ j;
            if(l > i){
                // COMPARE
                if((i & k) == 0 && (v[idx] > v[l-off]) || (i & k) != 0 && (v[idx] < v[l-off])){
                    // SWAP
                    tmp = v[idx];
                    v[idx] = v[l-off];
                    v[l-off] = tmp;
                }
            }
        }
        j /= 2;

        // Wait until the shared vector is completely rearranged before proceeding to the next step.
        // It is not necessary to synch at block level given that the following operations are local.
        __syncthreads(); 
    }

    // Write back the data from shared to global memory
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }    
}


__global__ void bitonicOffMem(int *a, int n, int k, int j, int elems_block){
/*
    Performs one step of bitonic sort. 
    In particular, it computes step (k, j).

    In these kernel all the computations are performed using the global memory.
*/ 

    // Block offset
    int off = blockIdx.x * elems_block;

    // One step of bitonic sort
    int l, i;
    int el_i, el_l;

    // Each thread evaluates comparisons and swaps until all elements are re-ordered. 
    // (according to the pattern required at step (k, j))
    for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
        i = idx + off;
        l = i ^ j;
        el_i = a[i];
        el_l = a[l];
        if(l > i){
            // COMPARE
            if((i & k) == 0 && (el_i > el_l) || (i & k) != 0 && (el_i < el_l)){
                // SWAP
                a[i] = el_l;
                a[l] = el_i;
            }
        }
    }
}


void bitonicSortPar(int *a, int n){
/*
    Bitonic sort is a sorting algorithm with log_2(n) steps each one containing 'i' stages,
    where 'i' is the index of the box.
    
    Advantages:
        - All the computations in the same stage can be done in parallel.
        - The workload across threads is balanced, as operations involve only comparisons and swaps.
            -> No huge delays when threads are synchronized
        - Parallel time complexity: θ( (n/p) * log(n)^2 )

    Disadvantages:
        - The number of elements in the input array needs to be a power of two.
*/

    // Compute the nearest power of two >= n
    int m = (int)pow(2, ceil(log2(n)));
    const size_t size = m * sizeof(int);

    // Instantiate an auxiliary array with a number of elements which is a power of two 
    int *aux, *d_aux;
    aux = (int *)malloc(size);              // Allocate space in HOST memory
    cudaMalloc((void **)&d_aux, size);      // Allocate space in DEVICE memory

    // Copy values into the auxiliary array
    for(int i=0; i < m; i++){
        if(i < n){
            aux[i] = a[i];
        } else {
            aux[i] = -1;
        }
    }

    // Copy the auxiliary array into the DEVICE 
    cudaMemcpy(d_aux, aux, size, cudaMemcpyHostToDevice);
    
    // Use the number of blocks that maximize the utilization of shared memory
    int blocks = (m + double(MAX_SHARED_ELEMS) - 1) / double(MAX_SHARED_ELEMS);
    int elems_block = (m + blocks - 1) / blocks;    // Number of elements per block
    int k=2;
    int j;

    // Perform initial re-arranging of the array up to k=elems_block
    if(elems_block <= MAX_SHARED_ELEMS){
        bitonicInit<<<blocks, 1024>>>(d_aux, m, elems_block);
        cudaDeviceSynchronize(); // Synch blocks
        k = elems_block;
    }
    
    // Continue re-arranging
    while(k <= m){
        j = k/2;
        while(j > 0){
            if(elems_block < MAX_SHARED_ELEMS && 2*j <= elems_block){
                // Dependencies between data on the same block only
                bitonicInMem<<<blocks, 1024>>>(d_aux, m, k, j, elems_block);
                j=0;
            } else {
                // Dependencies between data on different blocks
                bitonicOffMem<<<blocks, 1024>>>(d_aux, m, k, j, elems_block);
                j/=2;
            }
        }
        k*=2;
        cudaDeviceSynchronize(); // Synch blocks
    }
    
    // Copy the results from the array in the DEVICE to the one in the HOST memory
    cudaMemcpy(aux, d_aux, size, cudaMemcpyDeviceToHost);

    int i = 0;
    j = 0;
    while(i < m){
        if(aux[i] != -1){
            a[j] = aux[i];
            j++;
        }
        i++;
    }
    free(aux); cudaFree(d_aux);
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
                bitonicSortSeq(a, length);
                stop = clock();
            }

            timeS[r] = double(stop-start+1) / double(CLOCKS_PER_SEC);
            thrS[r] = double(length * (double(CLOCKS_PER_SEC) / double(stop-start+1)));

            // PARALLEL algorithm
            start = clock();
            bitonicSortPar(b, length);
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
