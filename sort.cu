#include <stdio.h>
#include <time.h>

#define N 10

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void init_array(int *a, int n){
    for(int i=0; i < n; i++){
        a[i] = rand();
    }
}

void copy_array(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        b[i] = a[i];
    } 
}

void selection_sort(int *a, int n){
    int tmp;
    for(int j=0; j < n; j++){
        for(int i=1; i < n-j; i++){
            if(a[i-1] > a[i]){
                tmp = a[i];
                a[i] = a[i-1];
                a[i-1] = tmp;
            }
        }
    }
}

void shell_sort(int *a, int n, int p){
    int tmp;
    for(int gap=p; gap > 0; gap /= 1.5){
        clock_t start = clock();
        // Check all the pairs
        for(int i=gap; i < n; i++){
            // Sort the current pair
            if(a[i-gap] > a[i]){
                tmp = a[i];
                a[i] = a[i-gap];
                a[i-gap] = tmp;
                // Sort previous terms
                for(int j=i-gap; j-gap >= 0 && a[j-gap] > a[j]; j=j-gap){
                    tmp = a[j];
                    a[j] = a[j-gap];
                    a[j-gap] = tmp;
                }
            }
        }
        clock_t stop = clock();
        printf("GAP %d time: %lf\n", gap, (double)(stop-start) / CLOCKS_PER_SEC);
    }
}


__global__ void parallel_transposition_sort(int *a, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    // Create shared sub-vector
    __shared__ int v[48/4 * 1024];

    // Place sub-array into shared memory
    if(idx < n){
        v[threadIdx.x] = a[idx];
    }

    int tmp;
    int phase = 0;
    if(threadIdx.x < blockDim.x-1 && idx < n-1){
        for(int i=0; i < blockDim.x; i++){
            if(threadIdx.x % 2 == phase % 2){
                if(v[threadIdx.x] > v[threadIdx.x+1]){
                    tmp = v[threadIdx.x];
                    v[threadIdx.x] = v[threadIdx.x+1];
                    v[threadIdx.x+1] = tmp;
                }
            }
            phase = 1-phase;
            __syncthreads();        
        }
    }
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("\n");
    }
    if(blockIdx.x == 0){
        printf("%d ", v[threadIdx.x]);
    }


    if(idx < n){
        a[idx] = v[threadIdx.x];
    }
}


__global__ void parallel_shell_sort(int *a, int n, int elems_per_block){

    // Create shared sub-vector
    __shared__ int v[12 * 1024];

    // Place sub-array into shared memory
    if(threadIdx.x == 0){
        for(int i=blockIdx.x*elems_per_block; i < n && i < (blockIdx.x+1)*elems_per_block; i++){
            v[i-blockIdx.x*elems_per_block] = a[i];
        }
    }

    //SHELL SORT!!!
}


void sort(int *a, int n){
    int *d_a;
    int size = n * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int blocks = (n + (12 * 1024) - 1) / (12 * 1024);
    int elems_per_block;

    while(true){
        elems_per_block = (n + blocks - 1) / blocks;
        if(elems_per_block > 1024){
            /* Launch sort() kernel on GPU */
            parallel_shell_sort<<<blocks, 1024>>>(d_a, n, elems_per_block);
        } else {
            /* Launch sort() kernel on GPU */
            parallel_transposition_sort<<<blocks, elems_per_block>>>(d_a, n);
        }

        if(blocks == 1) break;
        blocks = blocks / 2;
    }








    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
}




int main(){

    int *a;
    const size_t size = N * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    //cudaMalloc((void **)&d_a, size);
    
    /* Alloc space for host copies of a,b,c and setup input values */
    a = (int *)malloc(size);

    // Init Array
    for(int i=0; i < N; i++){
        a[i] = rand() % 10000;
        printf("%d ", a[i]);
    }
    printf("\n");

    sort(a, N);

    /*
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Launch sort() kernel on GPU
    parallel_shell_sort<<<4, 5>>>(d_a, N);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    */

    printf("\n");
    for(int i=0; i < N; i++){
        printf("%d ", a[i]);
    }
    printf("\n");

}


/*
int main() {
    
    clock_t start, stop;
    int a[N], b[N];
    bool good = true;

    init_array(a, N);
    copy_array(a, b, N);

    selection_sort(a, N);
    start = clock();
    shell_sort(b, N, 1);
    stop = clock();
    printf("%lf\n", (double)(stop-start) / CLOCKS_PER_SEC);

    for(int i=0; i < N; i++){
        printf("%d %d\n", a[i], b[i]);
        if(a[i] != b[i]){
            good = false;
            break;
        }
    }
    if(good){
        printf("GOOD!!");
    } else {
        printf("BAD!!");
    }
    
    //cuda_hello<<<1,1>>>(); 
    return 0;
}
*/