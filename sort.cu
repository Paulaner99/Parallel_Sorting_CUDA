#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 8*1024
#define N 128*8*1024

__global__ void sort_in_mem(int *a, int n, int k, int j, int blocks){
    int elems_block = (n + blocks - 1) / blocks;
    int off = blockIdx.x * elems_block;

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS];

    // Fill shared memory: each block loads the values in a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }
    __syncthreads();

    while(j > 0){
        // One step of bitonic sort
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
        j /= 2;
        __syncthreads();
    }

    // Write back the data from shared to global memory
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }    
}


__global__ void sort_out_mem(int *a, int n, int k, int j, int blocks){
    int elems_block = (n + blocks - 1) / blocks;
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


__global__ void sort_init(int *a, int n, int blocks){
    int elems_block = (n + blocks - 1) / blocks;
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
    int *d_a;
    int size = n * sizeof(int);
    int elems_block = (n + blocks - 1) / blocks;

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int k=2;
    int j;
    if(elems_block <= MAX_SHARED_ELEMS){
        printf("here");
        sort_init<<<blocks, threads>>>(d_a, n, blocks);
        k = elems_block;
    }
    
    cudaDeviceSynchronize();
    while(k <= n){
        j = k/2;
        while(j > 0){
            if(elems_block < MAX_SHARED_ELEMS && 2*j <= MAX_SHARED_ELEMS){
                sort_in_mem<<<blocks, threads>>>(d_a, n, k, j, blocks);
                j=0;
            } else {
                sort_out_mem<<<blocks, threads>>>(d_a, n, k, j, blocks);
                j/=2;
            }
        }
        k*=2;
    }
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
}


int main(){

    int *a, *aux;
    const size_t size = N * sizeof(int);

    a = (int *)malloc(size);

    // Init Array
    for(int i=0; i < N; i++){
        a[i] = rand();
        //printf("%d ", a[i]);
    }

    int max_pow_2 = (int)pow(2, (int)log2(N));
    int n;
    if(max_pow_2 != N){
        aux = (int *)malloc(2*max_pow_2*sizeof(int));
        n = 2*max_pow_2;
    } else {
        aux = (int *)malloc(size);
        n = N;
    }

    for(int i=0; i < n; i++){
        if(i < N){
            aux[i] = a[i];
        } else {
            aux[i] = -1;
        }
    }

    printf("\nStarted...\n");
    clock_t start = clock();
    //bitonic_seq(aux, n); 
    bitonic(aux, n, 128, 1024);
    clock_t stop = clock();
    printf("Finished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    printf("\n");
    for(int i = 0; i < n; i++){
        //printf("%d ", aux[i]);
    }
        
}