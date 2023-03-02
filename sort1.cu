#include <stdio.h>
#include <time.h>

#define N 1024*12*1024

void copy_array(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        b[i] = a[i];
    } 
}

void simple_sort(int *a, int n){
    int tmp;
    for(int j=1; j < n; j++){
        for(int i=0; i < n-j; i++){
            if(a[i] > a[i+1]){
                tmp = a[i];
                a[i] = a[i+1];
                a[i+1] = tmp;
            }
        }
    }
}

void shell_sort(int *a, int n){
    int tmp;
    for(int gap=1024; gap > 0; gap /= 2){
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
    }
}

void merge_seq(int *a, int n, int off, int elems){
    int *aux = (int *)malloc(n * sizeof(int));

    int i = 0;
    int j = 0;
    while(i < elems && j < elems && i + off < n && j + off + elems < n){
        if(a[off+i] < a[off+elems+j]){
            aux[off + i + j] = a[off+i];
            i++;
        } else {
            aux[off + i + j] = a[off+elems+j];
            j++;
        }
    }

    while(i < elems && i + off < n){
        aux[off + i + j] = a[off+i];
        i++;
    }

    while(j < elems && j + off + elems < n){
        aux[off + i + j] = a[off+elems+j];
        j++;
    }

    for(i = off; i < off + 2*elems && i < n; i++){
        a[i] = aux[i];
    }
}

void sort_seq(int *a, int n){

    int blocks = (n + (12 * 1024) - 1) / (12 * 1024);
    int elems = n / blocks; // elems < 12*1024

    int tmp;
    for(int block=0; block < blocks; block++){
        for(int gap=1024; gap > 0; gap /= 2){
            for(int i=block*elems+gap; i < block*elems + elems && i < n; i++){
                // Sort the current pair
                if(a[i-gap] > a[i]){
                    tmp = a[i];
                    a[i] = a[i-gap];
                    a[i-gap] = tmp;
                    // Sort previous terms
                    for(int j=i-gap; j-gap >= block*elems && a[j-gap] > a[j]; j=j-gap){
                        tmp = a[j];
                        a[j] = a[j-gap];
                        a[j-gap] = tmp;
                    }
                }
            }
        }
    }
    
    while(blocks > 1){
        blocks = (blocks+1) / 2; // ceil division by 2
        for(int block=0; block < blocks; block++){
            merge_seq(a, n, 2*block*elems, elems);
        }
        elems *= 2;
    }

}

__global__ void parallel_shell_sort(int *a, int n, int elems_per_block){
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
                for(int i=gap+threadIdx.x; i < elems_per_block && i+off < n; i=i+gap){
                    if(a[i+off-gap] > a[i+off]){
                        tmp = a[i+off];
                        a[i+off] = a[i+off-gap];
                        a[i+off-gap] = tmp;  
                    
                        for(int j=i-gap; j-gap >= off && a[j+off-gap] > a[j+off]; j=j-gap){
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
    int off = 2 * elems_per_block * blockIdx.x;
    
    int i = 0;
    int j = 0;
    while(i < elems_per_block && j < elems_per_block && i + off < n && j + off + elems_per_block < n){
        if(a[off+i] < a[off+elems_per_block+j]){
            aux[off + i + j] = a[off+i];
            i++;
        } else {
            aux[off + i + j] = a[off+elems_per_block+j];
            j++;
        }    
    }

    while(i < elems_per_block && i + off < n){
        aux[off + i + j] = a[off+i];
        i++;
    }

    while(j < elems_per_block && j + off + elems_per_block < n){
        aux[off + i + j] = a[off+elems_per_block+j];
        j++;
    }

    for(i = off; i < off + 2*elems_per_block && i < n; i++){
        a[i] = aux[i];
    }
}

void sort_par(int *a, int n){
    int *d_a, *d_aux;
    int size = n * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_aux, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, a, size, cudaMemcpyHostToDevice);

    int blocks = (n + (12 * 1024) - 1) / (12 * 1024);
    int elems_per_block = n / blocks;

    clock_t start = clock();
    clock_t stop;

    // Shell sort on sub-arrays
    parallel_shell_sort<<<blocks, 1024>>>(d_a, n, elems_per_block);
    cudaDeviceSynchronize();
    
    stop = clock();
    printf("Finished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);
    
    // Merge sub-arrays
    start = clock();
    int step = 1;
    while(blocks > 1){
        start = clock();
        blocks = (blocks+1) / 2; // ceil division by 2
        parallel_merge<<<blocks, 1>>>(d_a, d_aux, n, elems_per_block);
        cudaDeviceSynchronize();
        elems_per_block *= 2;
        stop = clock();
        //printf("Finished merge step %d in %lf seconds\n", step, (double)(stop-start) / CLOCKS_PER_SEC);
        step++;
    }
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

}

int main(){

    int *a, *b;
    const size_t size = N * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    //cudaMalloc((void **)&d_a, size);
    
    /* Alloc space for host copies of a,b,c and setup input values */
    a = (int *)malloc(size);
    b = (int *)malloc(size);

    // Init Array
    for(int i=0; i < N; i++){
        a[i] = rand();
        //printf("%d ", a[i]);
    }
    printf("\n");

    copy_array(a, b, N);

    clock_t start = clock();
    sort_seq(b, N);
    clock_t stop = clock();
    printf("\nTotal time (sequential): %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    start = clock();
    sort_par(a, N);
    stop = clock();
    printf("\nTotal time (parallel): %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    bool equal = true;
    for(int i=0; i < N; i++){
        if(a[i] != b[i]){
            equal = false;
            break;
        }
    }

    //for(int i=0; i<N; i++)
    //    printf("%d ", a[i]);

    if(equal){
        printf("\nWell done!");
    } else {
        printf("\nThere's some error!");
    }
}