#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 2*12*1024

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

void merge_seq(int *a, int n, int off, int elems_per_block){
    int *aux = (int *)malloc(n * sizeof(int));
    int i = 0;
    int j = 0;
    int off_1 = off;
    int off_2 = off + elems_per_block;
    while(i + off_1 < off_2 && j + off_2 < off_2 + elems_per_block && i + off_1 < n && j + off_2 < n){
        if(a[i] < a[j]){
            aux[off_1 + i + j] = a[off_1+i];
            i++;
        } else {
            aux[off_1 + i + j] = a[off_2+j];
            j++;
        }    
    }

    while(i + off_1 < off_2 && i + off_1 < n){
        aux[off_1 + i + j] = a[off_1+i];
        i++;
    }

    while(j + off_2 < off_2 + elems_per_block && j + off_2 < n){
        aux[off_1 + i + j] = a[off_2+j];
        j++;
    }

    for(i = off_1; i < off_2 + elems_per_block && i < n; i++){
        a[i] = aux[i];
    }
}

void sort_seq(int *a, int n){

    int blocks = (n + (12 * 1024) - 1) / (12 * 1024);
    int elems_per_block = (n + blocks - 1) / blocks;

    int tmp;
    for(int block=0; block < blocks; block++){
        for(int gap=1024; gap > 0; gap /= 2){
            for(int i=block*elems_per_block+gap; i < (block+1)*elems_per_block && i < n; i++){
                // Sort the current pair
                if(a[i-gap] > a[i]){
                    tmp = a[i];
                    a[i] = a[i-gap];
                    a[i-gap] = tmp;
                    // Sort previous terms
                    for(int j=i-gap; j-gap >= block*elems_per_block+gap && a[j-gap] > a[j]; j=j-gap){
                        tmp = a[j];
                        a[j] = a[j-gap];
                        a[j-gap] = tmp;
                    }
                }
            }
        }
    }

    for(int step=2; step <= blocks; step*=2){
        for(int block=0; block < blocks / step; block=block+2){
            merge_seq(a, n, block*elems_per_block, elems_per_block);
        }
        elems_per_block *= 2;
    }
    for(int i=0; i < n; i++){
        printf("%d ", a[i]);
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
    __shared__ int v[1024];

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

    if(idx < n){
        a[idx] = v[threadIdx.x];
    }
}


__global__ void parallel_shell_sort(int *a, int n, int elems_per_block){
    int blocks = (n + elems_per_block - 1) / elems_per_block;

    //Block offset
    int off = blockIdx.x*elems_per_block;

        
    // Create shared sub-vector
    __shared__ int v[12 * 1024];

    // Load elements from the original array to the shared memory
    if(elems_per_block < 1024){
        if(threadIdx.x < elems_per_block && threadIdx.x+off < n){
            v[threadIdx.x] = a[threadIdx.x+off];
        }
    } else {
        for(int i=threadIdx.x; i < elems_per_block && i+off < n; i=i+1024){
            v[i] = a[i+off];
        }
    }

    // Shell sort
    int tmp;
    for(int gap=1024; gap > 0; gap /= 2){
        if(threadIdx.x < gap){
            for(int i=gap+threadIdx.x; i < elems_per_block; i=i+gap){
                if(v[i-gap] > v[i]){
                    tmp = v[i];
                    v[i] = v[i-gap];
                    v[i-gap] = tmp;  
                
                    for(int j=i-gap; j-gap >= 0 && v[j-gap] > v[j]; j=j-gap){
                        tmp = v[j];
                        v[j] = v[j-gap];
                        v[j-gap] = tmp;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Save elements from the shared memory to the original array
    if(elems_per_block < 1024){
        if(threadIdx.x < elems_per_block && threadIdx.x+off < n){
            a[threadIdx.x+off] = v[threadIdx.x];
        }
    } else {
        for(int i=threadIdx.x; i < elems_per_block && i+off < n; i=i+1024){
            a[i+off] = v[i];
        }
    }
}

__global__ void parallel_shell_sort_mix(int *a, int n, int elems_per_block){
    int blocks = (n + elems_per_block - 1) / elems_per_block;

    //Block offset
    int off = blockIdx.x*elems_per_block;
        
    // Create shared sub-vector
    __shared__ int v[12 * 1024];

    // Load elements from the original array to the shared memory
    if(elems_per_block < 1024){
        if(threadIdx.x < elems_per_block && threadIdx.x+off < n){
            v[threadIdx.x] = a[threadIdx.x+off];
        }
    } else {
        for(int i=threadIdx.x; i < elems_per_block && i+off < n; i=i+1024){
            v[i] = a[i+off];
        }
    }

    // Shell sort
    int tmp;
    for(int gap=1024; gap > 0; gap /= 2){
        if(threadIdx.x < gap){
            for(int i=gap+threadIdx.x; i < elems_per_block; i=i+gap){
                if(v[i-gap] > v[i]){
                    tmp = v[i];
                    v[i] = v[i-gap];
                    v[i-gap] = tmp;  
                
                    for(int j=i-gap; j-gap >= 0 && v[j-gap] > v[j]; j=j-gap){
                        tmp = v[j];
                        v[j] = v[j-gap];
                        v[j-gap] = tmp;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Save elements from the shared memory to the original array
    if(threadIdx.x == 0){
        for(int i=0; i < elems_per_block && i+off < n; i++){
            a[i*blocks+blockIdx.x] = v[i];
        }
    }
}

__global__ void parallel_shell_sort_new(int *a, int n, int elems_per_block){
    int blocks = (n + elems_per_block - 1) / elems_per_block;

    //Block offset
    int off = blockIdx.x*elems_per_block;

    if(elems_per_block <= 12*1024){
    
        // Create shared sub-vector
        __shared__ int v[12 * 1024];

        // Load elements from the original array to the shared memory
        if(elems_per_block < 1024){
            if(threadIdx.x < elems_per_block && threadIdx.x+off < n){
                v[threadIdx.x] = a[threadIdx.x+off];
            }
        } else {
            for(int i=threadIdx.x; i < elems_per_block && i+off < n; i=i+1024){
                v[i] = a[i+off];
            }
        }

        // Shell sort
        int tmp;
        for(int gap=1024; gap > 0; gap /= 2){
            if(threadIdx.x < gap){
                for(int i=gap+threadIdx.x; i < elems_per_block; i=i+gap){
                    if(v[i-gap] > v[i]){
                        tmp = v[i];
                        v[i] = v[i-gap];
                        v[i-gap] = tmp;  
                    
                        for(int j=i-gap; j-gap >= 0 && v[j-gap] > v[j]; j=j-gap){
                            tmp = v[j];
                            v[j] = v[j-gap];
                            v[j-gap] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Save elements from the shared memory to the original array
        if(elems_per_block < 1024){
            if(threadIdx.x < elems_per_block && threadIdx.x+off < n){
                a[threadIdx.x+off] = v[threadIdx.x];
            }
        } else {
            for(int i=threadIdx.x; i < elems_per_block && i+off < n; i=i+1024){
                a[i+off] = v[i];
            }
        }
    } else {
        // Shell sort
        int tmp;
        for(int gap=1024; gap > 0; gap /= 2){
            if(threadIdx.x < gap){
                for(int i=gap+threadIdx.x; i < elems_per_block; i=i+gap){
                    if(a[i+off-gap] > a[i+off]){
                        tmp = a[i+off];
                        a[i+off] = a[i+off-gap];
                        a[i+off-gap] = tmp;  
                    
                        for(int j=i-gap; j-gap >= 0 && a[j+off-gap] > a[j+off]; j=j-gap){
                            tmp = a[j+off];
                            a[j+off] = a[j+off-gap];
                            a[j+off-gap] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void parallel_merge(int *a, int *aux, int n, int elems_per_block){
    int off_1 = elems_per_block * blockIdx.x;
    int off_2 = elems_per_block * (blockIdx.x+1);
    
    int i = 0;
    int j = 0;
    while(i + off_1 < off_2 && j + off_2 < elems_per_block * (blockIdx.x+2) && j + off_2 < n){
        if(a[i] < a[j]){
            aux[off_1 + i + j] = a[off_1+i];
            i++;
        } else {
            aux[off_1 + i + j] = a[off_2+j];
            j++;
        }    
    }

    while(i + off_1 < off_2 && i + off_1 < n){
        aux[off_1 + i + j] = a[off_1+i];
        i++;
    }

    while(j + off_2 < elems_per_block * (blockIdx.x+2) && j + off_2 < n){
        aux[off_1 + i + j] = a[off_2+j];
        j++;
    }

    for(i = off_1; i < elems_per_block * (blockIdx.x+2) && i < n; i++){
        a[i] = aux[i];
    }
    //printf("\n\n\nBLOCK: %d; START: %d; END: %d\n\n\n", blockIdx.x, off_1, off_2);
}


void sort(int *a, int n){
    int *d_a, *d_aux;
    int size = n * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_aux, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, a, size, cudaMemcpyHostToDevice);

    int blocks = (n + (12 * 1024) - 1) / (12 * 1024);
    int elems_per_block = (n + blocks - 1) / blocks;
    printf("\n%d\n", blocks);

    clock_t start = clock();
    clock_t stop;

    // Shell sort on sub-arrays
    parallel_shell_sort<<<blocks, 1024>>>(d_a, n, elems_per_block);
    cudaDeviceSynchronize();
    
    stop = clock();
    printf("Finished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);
    
    // Merge sub-arrays
    printf("\n\n");
    start = clock();
    int step = 1;
    while(blocks > 1){
        start = clock();
        blocks /= 2;
        elems_per_block = (n + blocks - 1) / blocks;
        parallel_merge<<<blocks, 1>>>(d_a, d_aux, n, elems_per_block);
        cudaDeviceSynchronize();
        stop = clock();
        printf("Finished merge step %d in %lf seconds\n", step, (double)(stop-start) / CLOCKS_PER_SEC);
        step++;
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
        a[i] = rand();
        //printf("%d ", a[i]);
    }
    printf("\n");

    sort_seq(a, N);

    //printf("STARTED...\n");
    clock_t start = clock();
    sort(a, N);
    clock_t stop = clock();
    printf("\nTotal time: %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    //printf("STARTED...\n");
    //start = clock();
    //quickSort(a, N);
    //stop = clock();
    //printf("Finished in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);
    /*
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Launch sort() kernel on GPU
    parallel_shell_sort<<<4, 5>>>(d_a, N);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    
    
    printf("\n");
    for(int i=0; i < N; i++){
        printf("%d ", a[i]);
    }
    */
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