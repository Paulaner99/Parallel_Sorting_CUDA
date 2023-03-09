#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 512*1024*1024

__global__ void sort(int *a, int n, int k, int j, int blocks){
    int elems_block = n / blocks;
    int off = blockIdx.x * elems_block;

    if(elems_block < 8 *1024 && k <= elems_block){
        // Shared memory
        __shared__ int v[8 * 1024];

        // Fill shared memory: each block loads the values in a local vector 'v'
        for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
            v[i] = a[i+off];
        }
        __syncthreads();

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
        
        // Write back the data from shared to global memory
        for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
            a[i+off] = v[i];
        }    

    } else {
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
}

void bitonic(int *a, int n, int blocks, int threads){
    int *d_a;
    int size = n * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    for(int k=2; k <= n; k*=2){
        for(int j=k/2; j > 0; j/=2){
            sort<<<blocks, threads>>>(d_a, n, k, j, blocks);
            cudaDeviceSynchronize();
        }
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
    bitonic(aux, n, 1024, 1024);
    clock_t stop = clock();
    printf("Finished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    printf("\n");
    //for(int i = 0; i < n; i++){
    //    printf("%d ", aux[i]);
    //}
        
}