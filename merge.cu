#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SHARED_ELEMS 8*1024
#define N 4*1024*1024
#define MAX_LENGTH 64*1024*1024


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


bool compare_arrays(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        if(a[i] != b[i]){
            return false;
        }
    }
    return true;
}


void mergeSeq(int *a, int *aux, int l, int m, int r){
    int n1 = m - l;
    int n2 = r - m;

    for(int i=0; i < n1+n2; i++){
        aux[l + i] = a[l + i];
    }

    int i = 0;
    int j = 0;
    int k = l;
    while(i < n1 && j < n2){
        if(aux[l + i] <= aux[m + j]){
            a[k] = aux[l + i];
            i++;
        } else {
            a[k] = aux[m + j];
            j++;
        }
        k++;
    }
  
    while(i < n1){
        a[k] = aux[l + i];
        i++;
        k++;
    }
  
    while(j < n2){
        a[k] = aux[m + j];
        j++;
        k++;
    }
}


// Shell sort
void shellSortSeq(int *a, int n, int interval){
    int blocks = (n + double(MAX_SHARED_ELEMS) - 1) / double(MAX_SHARED_ELEMS);
    int elems_block = ((n + blocks - 1) / blocks);

    int *aux = (int *)malloc(n * sizeof(int));
    clock_t start, stop;

    for(int block=0; block < blocks; block++){
        for(int gap=elems_block/2; gap > 0; gap/=2){
            for(int i=block*elems_block+gap; i < (block + 1) * elems_block && i < n; i++){
                //if(block == 0)
                //    printf("%d ", a[i]);
                int temp = a[i];
                int j;
                for(j=i; j >= block*elems_block+gap && a[j - gap] > temp; j -= gap){
                    a[j] = a[j - gap];
                    //if(block == 0)
                    //    printf("%d ", a[i]);
                }
                a[j] = temp;
            }
        }
    }
    
    start = clock();
    for(int i=blocks; i > 1; i/=2){
        elems_block = ((n + i - 1) / i);
        for(int j=0; j < i; j+=2){
            int l = j * elems_block;
            int m = l + elems_block;
            int r = min(m + elems_block, n);
            //printf("\n%d %d %d (%d)\n", l, m, r, elems_block);
            if(m < r){
                mergeSeq(a, aux, l, m, r);
            }
        }
    }
    stop = clock();
    printf("Merge time seq:\t%lf seconds\t", double(stop-start+1) / double(CLOCKS_PER_SEC));
}


__global__ void shellSort(int *a, int n, int interval){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < interval){
        for(int i=interval+idx; i < n; i+=interval){
            int tmp = a[i];
            int j;
            for(j=i; j >= interval && a[j - interval] > tmp; j-=interval){
                a[j] = a[j - interval];
            }
            a[j] = tmp;
        }
    }

}

__global__ void shellSortShared(int *a, int n, int blocks){
    int elems_block = (n + blocks - 1) / blocks;
    int off = elems_block * blockIdx.x;
    
    __shared__ int v[MAX_SHARED_ELEMS];

    // Fill shared memory: each block loads the values in a local vector 'v'
    for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
        v[i] = a[i + off];
    }
    
	__syncthreads(); // Wait for the local vector to be filled

    for(int interval=blockDim.x; interval > 0; interval/=2){
        if(threadIdx.x < interval){
            for(int i=interval+threadIdx.x; i < elems_block && i + off < n; i+=interval){
                //if(blockIdx.x == 0)
                //    printf("%d ", v[i]);
                int tmp = v[i];
                int j;
                for(j=i; j >= interval && v[j - interval] > tmp; j-=interval){
                    v[j] = v[j - interval];
                    //if(blockIdx.x == 1)
                    //    printf("%d ", v[j]);
                }
                v[j] = tmp;
            }
        }
        __syncthreads();
    }

    __syncthreads();
    for(int i=threadIdx.x; i < elems_block && i + off < n; i+=blockDim.x){
        a[i + off] = v[i];
    }
}


__device__ void mergePar(int *a, int *aux, int l, int m, int r){
    int n1 = m - l;
    int n2 = r - m;

    for(int i=0; i < n1+n2; i++){
        aux[l + i] = a[l + i];
        //printf("%d ", l + i);
    }

    int i = 0;
    int j = 0;
    int k = l;
    while(i < n1 && j < n2){
        if(aux[l + i] <= aux[m + j]){
            //printf("%d %d\t", aux[l + i], aux[m + j]);
            a[k] = aux[l + i];
            i++;
        } else {
            // NEVER ENTERS HERE
            //printf("%d %d\t", aux[l + i], aux[m + j]);
            a[k] = aux[m + j];
            j++;
        }
        k++;
    }
  
    while(i < n1){
        a[k] = aux[l + i];
        i++;
        k++;
    }
  
    while(j < n2){
        a[k] = aux[m + j];
        j++;
        k++;
    }
}


__global__ void mergeSort(int *a, int *aux, int n, int blocks){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int elems_block = (n + blocks - 1) / blocks;
    int l = idx * elems_block * 2;
    int m = l + elems_block;
    int r = min(l + 2 * elems_block, n);
    
    if(m < r){
        //printf("\n%d %d (%d): %d %d %d (%d)\n", threadIdx.x, blockIdx.x, blockDim.x + blockIdx.x, l, m, r, elems_block);
        mergePar(a, aux, l, m, r);
    }
}


void shellSortPar(int *a, int n){
/*
*/

    // Instantiate an auxiliary array with a number of elements which is a number of two 
    int *d_a;
	const size_t size = n * sizeof(int);
    clock_t start, stop;
    cudaMalloc((void **)&d_a, size);      // Allocate space in DEVICE memory
    int *aux = (int *)malloc(n * sizeof(int));      // Allocate space in HOST memory

    // Copy the auxiliary array into the DEVICE 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int blocks = (n + double(MAX_SHARED_ELEMS) - 1) / double(MAX_SHARED_ELEMS);
    int threads = 1024;
    
    shellSortShared<<<blocks, threads>>>(d_a, n, blocks);
    cudaDeviceSynchronize();
   
    // Copy the results from the array in the DEVICE to the one in the HOST memory
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    
    start = clock();
    for(int i=blocks; i > 1; i/=2){
        int elems_block = ((n + i - 1) / i);
        for(int j=0; j < i; j+=2){
            int l = j * elems_block;
            int m = l + elems_block;
            int r = min(m + elems_block, n);
            //printf("\n%d %d %d (%d)\n", l, m, r, elems_block);
            if(m < r){
                mergeSeq(a, aux, l, m, r);
            }
        }
    }
    stop = clock();
    printf("Merge time par:\t%lf seconds\t", double(stop-start+1) / double(CLOCKS_PER_SEC));

	printf("\n");
	//for(int i=0; i < n; i++){
		//printf("%d ", a[i]);
	//}
	printf("\n");

    cudaFree(d_a);
}

bool test(int *a, int *b, int n){
    clock_t start, stop;
    double time_seq, time_par, thr_seq, thr_par;

    printf("\nnum elements: %d\n", n);
    /* SEQUENTIAL BITONIC SORT */
    start = clock();
    shellSortSeq(a, n, n/2);
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

    for(int length=1024*8*1024; length < MAX_LENGTH; length*=2){
        // Allocate memory for the array
        a = (int *)malloc(length * sizeof(int));
        b = (int *)malloc(length * sizeof(int));

        // Initialize random array
        random_array(a, length);
        copy_array(a, b, length);

        test(a, b, length);
		break;
        free(a); free(b);
    }
}