#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 12*1024
#define N 4*1024*1024
#define MAX_LENGTH 128*1024*1024


void random_array(int *a, int n){
/*
    Randomly initialize the elements of an array given its size.
*/
    for(int i=0; i < n; i++){
        a[i] = rand();
		//printf("%d ", a[i]);
    }
}


void copy_array(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        b[i] = a[i];
    }
}


__global__ void copyArrayDevice(int *dst, int *src, int n, int blocks){
    int threads = blocks * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=idx; i < n; i+=threads){
        dst[i] = src[i];
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


__host__ __device__ int binarySearchCount(int val, int *a, int first, int last, int ord){
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


__global__ void merge_in_mem(int *a, int n, int blocks){
    int elems_block = (n + blocks - 1) / blocks; 
    
    // Block offset
    int off = blockIdx.x * elems_block;

    // Shared memory
    __shared__ int v[MAX_SHARED_ELEMS/2];
    __shared__ int aux[MAX_SHARED_ELEMS/2];

    // Fill shared memory: each block loads the values in a local vector 'v'
    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        v[i] = a[i+off];
    }

    for(int size=1; size < elems_block; size*=2){
        __syncthreads();
        for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
            int before = i%size + (i/(2*size)) * 2 * size;
            int ord = (i/size) % 2;
            int first = (i/size + 1 - 2*ord) * size;
            int last = min(first + size, n - off);
            if(last > first){
                int idx = binarySearchCount(v[i], v, first, last, ord);
                aux[idx + before] = v[i]; 
            } else {
                aux[before] = v[i];
            }
            
        }
        __syncthreads();
        for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
            v[i] = aux[i]; 
        }
    }

    for(int i=threadIdx.x; (i < elems_block) && (i + off < n); i+=blockDim.x){
        a[i+off] = v[i];
    }
}


__global__ void merge_out_mem(int *a, int *aux, int n, int size, int blocks){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blocks * blockDim.x;

    for(int i=idx; i < n; i+=threads){
        int before = i%size + (i/(2*size)) * 2 * size;
        int ord = (i/size) % 2;
        int first = (i/size + 1 - 2*ord) * size;
        int last = min(first + size, n);
        if(last > first){
            int idx = binarySearchCount(a[i], a, first, last, ord);
            aux[idx + before] = a[i];
            //printf("\n%d\t%d -> %d (%d + %d)\t%d %d (%d)\n", a[i], i, idx + before, idx, before, first, last, ord);
        } else {
            aux[before] = a[i];
            //printf("\n%d\t%d -> %d\t%d %d (%d)\n", a[i], before, first, last, ord);
        }
    }
}


void mergeSort(int *a, int n){
    int *aux = (int *)malloc(n * sizeof(int));
    for(int size=1; size < n; size*=2){
        for(int i=0; i < n; i++){
            int before = i%size + (i/(2*size)) * 2 * size;
            int ord = (i/size) % 2;
            int first = (i/size + 1 - 2*ord) * size;
            int last = min(first + size, n);
            if(last > first){
                int idx = binarySearchCount(a[i], a, first, last, ord);
                aux[idx + before] = a[i];
            } else {
                aux[before] = a[i];
            }
        }
        copy_array(aux, a, n);
    }
    printf("\n");
	for(int i=0; i < n; i++){
		//printf("%d ", a[i]);
	}
	printf("\n");
}


void shellSortPar(int *a, int n){
/*
*/

    // Instantiate an auxiliary array with a number of elements which is a number of two 
    int *d_a, *d_aux;
	const size_t size = n * sizeof(int);
    clock_t start, stop;
    cudaMalloc((void **)&d_a, size);      // Allocate space in DEVICE memory
    cudaMalloc((void **)&d_aux, size);    // Allocate space in DEVICE memory

    // Copy the auxiliary array into the DEVICE 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int blocks = (n + double(MAX_SHARED_ELEMS/2) - 1) / double(MAX_SHARED_ELEMS/2);
    int threads = 1024;
    int elems_block = (n + blocks - 1) / blocks;
    
    //merge_in_mem<<<blocks, threads>>>(d_a, n, blocks);
    cudaDeviceSynchronize();
    for(int size=1; size < n; size*=2){
        merge_out_mem<<<blocks, threads>>>(d_a, d_aux, n, size, blocks);
        cudaDeviceSynchronize();
        cudaMemcpy(d_a, d_aux, size, cudaMemcpyDeviceToDevice);
        copyArrayDevice<<<blocks, threads>>>(d_a, d_aux, n, blocks);
        cudaDeviceSynchronize();
    }
   
    // Copy the results from the array in the DEVICE to the one in the HOST memory
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    

	printf("\n");
	for(int i=0; i < n; i++){
		//printf("%d ", a[i]);
	}
	printf("\n");

    cudaFree(d_a);
}

bool test(int *a, int *b, int n){
    clock_t start, stop;
    double time_seq, time_par, thr_seq, thr_par;

    printf("\nnum elements: %d\n", n);
    /* SEQUENTIAL BITONIC SORT */
    start = clock();
    //mergeSort(a, n);
    stop = clock();
    time_seq = double(stop-start+1) / double(CLOCKS_PER_SEC);
    thr_seq = double(n * (double(CLOCKS_PER_SEC) / double(stop-start+1)));
    
    /* PARALLEL BITONIC SORT */
    start = clock();
    int blocks = int(floor((double)(n + (double)(MAX_SHARED_ELEMS / 2) - 1) / (double)(MAX_SHARED_ELEMS / 2)));
    shellSortPar(b, n);
    stop = clock();
    time_par = double(stop-start+1) / double(CLOCKS_PER_SEC);
    thr_par = double(n * (double(CLOCKS_PER_SEC) / double(stop-start+1)));

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

    for(int length=MAX_LENGTH; length <= MAX_LENGTH; length*=2){
        // Allocate memory for the array
        a = (int *)malloc(length * sizeof(int));
        b = (int *)malloc(length * sizeof(int));

        // Initialize random array
        random_array(a, length);
        copy_array(a, b, length);

        test(a, b, length);
		//break;
        free(a); free(b);
    }
}