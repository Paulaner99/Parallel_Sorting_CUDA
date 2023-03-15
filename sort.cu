#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 8*1024
#define N 256*1024*1024
#define MAX_LENGTH 256*1024*1024


void random_array(int *a, int n){
/*
    Randomly initialize the elements of an array given its size.
*/
    for(int i=0; i < N; i++){
        a[i] = rand();
    }
}


void copy_array(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        b[i] = a[i];
    }
}


bool compare_arrays(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        if(a[i] != b[i]){
            return false;
        }
    }
    return true;
}


void compare_swap(int *a, int i, int j, int dir){
    if(dir == (a[i] > a[j])){
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}


void bitonic_merge(int *a, int low, int k, int dir){
    if(k > 1){
        int j = k/2;
        for(int i=low; i < low + j; i++){
            compare_swap(a, i, i+j, dir);
        }
        bitonic_merge(a, low, j, dir);
        bitonic_merge(a, low+j, j, dir);
    }
}


void bitonic_sort(int *a,int low, int k, int dir){
    if(k > 1){
        int j = k/2;
 
        // Sort in ascending order (dir=1)
        bitonic_sort(a, low, j, 1);
 
        // Sort in descending order (dir=0)
        bitonic_sort(a, low+j, j, 0);
 
        // Merge whole sequence in ascending order (dir=1)
        bitonic_merge(a,low, k, dir);
    }
}


void bitonic_seq(int *a, int n){
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
    
    bitonic_sort(aux, 0, m, 1);

    int i = 0;
    int j = 0;
    while(i < m){
        if(aux[i] != -1){
            a[j] = aux[i];
            //printf("%d ", a[j]);
            j++;
        }
        i++;
    }
    free(aux);
}


__global__ void sort_in_mem(int *a, int n, int k, int j, int elems_block){
/*
    Performs multiple steps of bitonic sort within the same box 'k'. 
    In particular, it computes log_2(j) steps (i.e., up to j=1, with j halved at each step).

    The value of 'j' represents the 'distance' between elements to be compared (and eventually swapped).
        -> 2*j is the number of elements on which the following steps within the 'k' box depend!

    In these function all the computations are performed using the shared memory.
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


__global__ void sort_out_mem(int *a, int n, int k, int j, int elems_block){
/*
    Performs one step of bitonic sort. 
    In particular, it computes step (k, j).

    In these function all the computations are performed using the global memory.
*/ 

    // Block offset
    int off = blockIdx.x * elems_block;

    // One step of bitonic sort
    int l, i, tmp;

    // Each thread evaluates comparisons and swaps until all elements are re-ordered. 
    // (according to the pattern required at step (k, j))
    for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
        i = idx + off;
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


__global__ void sort_init(int *a, int n, int elems_block){
/*
    Performs multiple steps of bitonic sort until dependency between data remains within blocks. 
    In particular, it computes all the boxes up to k=elems_block. This is the last block with no external dependencies.

    In these function all the computations are performed using the shared memory.
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


void bitonic(int *a, int n, int blocks, int threads){
/*
    Bitonic sort is a sorting algorithm with log_2(n) boxes each one containing 'i' steps,
    where 'i' is the index of the box.
    
    Advantages:
        - All the computations in the same step can be done in parallel.
        - The workload across threads is balanced, as operations involve only comparisons and swaps.
            -> No huge delays when threads are synchronized
        - Parallel time complexity: θ( (n/p) * log(n)^2 )

    Disadvantages:
        - The number of elements in the input array needs to be a power of two.
*/

    // Compute the nearest power of two >= n
    int m = (int)pow(2, ceil(log2(n)));
    const size_t size = m * sizeof(int);

    // Instantiate an auxiliary array with a number of elements which is a number of two 
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
    
    int elems_block = (m + blocks - 1) / blocks;    // Number of elements per block
    int k=2;
    int j;

    // Perform initial re-arranging of the array up to k=elems_block
    if(elems_block <= MAX_SHARED_ELEMS){
        sort_init<<<blocks, threads>>>(d_aux, m, elems_block);
        cudaDeviceSynchronize(); // Synch blocks
        k = elems_block;
    }
    
    // Continue re-arranging
    while(k <= m){
        j = k/2;
        while(j > 0){
            if(elems_block < MAX_SHARED_ELEMS && 2*j <= elems_block){
                // Dependencies between data on the same block only
                sort_in_mem<<<blocks, threads>>>(d_aux, m, k, j, elems_block);
                j=0;
            } else {
                // Dependencies between data on different blocks
                sort_out_mem<<<blocks, threads>>>(d_aux, m, k, j, elems_block);
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
            //printf("%d ", a[j]);
            j++;
        }
        i++;
    }
    free(aux); cudaFree(d_aux);
}


bool test(int *a, int *b, int n){
    clock_t start, stop;
    double time_seq, time_par, thr_seq, thr_par;

    printf("\nnum elements: %d\n", n);
    /* SEQUENTIAL BITONIC SORT */
    start = clock();
    bitonic_seq(a, n);
    stop = clock();
    time_seq = (double)(stop-start) / CLOCKS_PER_SEC;
    thr_seq = (double) (n * ((double)(CLOCKS_PER_SEC) / (stop-start)));
    
    /* PARALLEL BITONIC SORT */
    start = clock();

    int blocks = int(floor((double)(n + (double)(MAX_SHARED_ELEMS) - 1) / (double)(MAX_SHARED_ELEMS)));
    bitonic(b, n, blocks, 1024);

    stop = clock();
    time_par = (double)(stop-start) / CLOCKS_PER_SEC;
    thr_par = (double) (n * ((double)(CLOCKS_PER_SEC) / (stop-start)));

    printf("Sequential time:\t%lf seconds\t", time_seq);
    printf("Throughput:\t%lf\n", thr_seq);
    printf("  Parallel time:\t%lf seconds\t", time_par);
    printf("Throughput:\t%lf\t(%d)\t", thr_par, blocks);
    if(compare_arrays(a, b, n)){
        printf("[OK]\n");
        return true;
    } else {
        printf("[WRONG]\n");
        return false;
    }
}

int main(){

    int *a, *b;
    const size_t size = N * sizeof(int);

    for(int length=1024; length < MAX_LENGTH; length*=2){
        // Allocate memory for the array
        a = (int *)malloc(size);
        b = (int *)malloc(size);

        // Initialize random array
        random_array(a, length);
        copy_array(a, b, length);

        test(a, b, length);
        free(a); free(b);
    }
}