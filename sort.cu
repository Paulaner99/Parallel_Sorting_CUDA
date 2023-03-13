#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 8*1024
#define N 32*8*256*1024

void random_array(int *a, int n){
/*
    Randomly initialize the elements of an array given its size.
*/
    for(int i=0; i < N; i++){
        a[i] = rand();
    }
}

__global__ void sort_in_mem(int *a, int n, int k, int j, int elems_block){
/*
    Performs multiple steps of bitonic sort within the same box 'k'. 
    In particular, it computes log_2(j) steps (i.e., up to j=1, with j halved at each step).

    The value of 'j' represents the 'distance' between elements to be compared (and eventually swapped).
        -> 2*j is the number of elements on which the following steps within the 'k' box depend!

    In these function all the computations are performed using the shared memory.
*/    

    int off = blockIdx.x * elems_block; // Block offset

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS];

    // Fill shared memory: each block loads values into a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }
    __syncthreads(); // Wait for the local vector to be filled

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
    int off = blockIdx.x * elems_block;

    // Global memory
    // One step of bitonic sort
    int l;
    int tmp;
    int i;
    for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
        i = idx + off;
        l = i ^ j;
        if(l > i){
            if((i & k) == 0 && (a[i] > a[l]) || (i & k) != 0 && (a[i] < a[l])){
                tmp = a[i];
                a[i] = a[l];
                a[l] = tmp;
            }
        }
    }
}


__global__ void sort_init(int *a, int n, int elems_block){
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
            int l;
            int tmp;
            int i;
            for(int idx=threadIdx.x; (idx < elems_block) && (idx + off < n); idx+=blockDim.x){
                i = idx + off;
                l = i ^ j;
                if(l > i){
                    if((i & k) == 0 && (v[idx] > v[l-off]) || (i & k) != 0 && (v[idx] < v[l-off])){
                        tmp = v[idx];
                        v[idx] = v[l-off];
                        v[l-off] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Write back the data from shared to global memory
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }    
}

void bitonic(int *a, int n, int blocks, int threads){
    int *aux, *d_aux;
    
    int m = (int)pow(2, ceil(log2(N)));
    const size_t size = m * sizeof(int);
    int elems_block = (m + blocks - 1) / blocks;

    aux = (int *)malloc(size);
    // Copy values into auxiliary array
    for(int i=0; i < m; i++){
        if(i < n){
            aux[i] = a[i];
        } else {
            aux[i] = -1;
        }
    }

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_aux, size);

    /* Copy inputs to device */
    cudaMemcpy(d_aux, aux, size, cudaMemcpyHostToDevice);
    
    int k=2;
    int j;
    if(elems_block <= MAX_SHARED_ELEMS){
        sort_init<<<blocks, threads>>>(d_aux, m, elems_block);
        k = elems_block;
    }
    
    cudaDeviceSynchronize();
    while(k <= m){
        j = k/2;
        while(j > 0){
            if(elems_block < MAX_SHARED_ELEMS && 2*j <= elems_block){
                sort_in_mem<<<blocks, threads>>>(d_aux, m, k, j, elems_block);
                j=0;
            } else {
                sort_out_mem<<<blocks, threads>>>(d_aux, m, k, j, elems_block);
                j/=2;
            }
        }
        k*=2;
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(aux, d_aux, size, cudaMemcpyDeviceToHost);

    k = 0;
    j = 0;
    while(k < m){
        if(aux[k] != -1){
            a[j] = aux[k];
            //printf("%d ", a[j]);
            j++;
        }
        k++;
    }
}


int main(){

    int *a;
    const size_t size = N * sizeof(int);

    // Allocate memory for the array
    a = (int *)malloc(size);

    // Initialize random array
    random_array(a, N);

    printf("\nStarted...\n");
    clock_t start = clock();

    // Parallel bitonic sort
    bitonic(a, N, 256, 1024);

    clock_t stop = clock();
    printf("\nFinished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    printf("\n");
        
}