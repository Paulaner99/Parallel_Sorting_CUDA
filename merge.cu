#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 11

void copy_array(int *a, int *b, int n){
    for(int i=0; i < n; i++){
        b[i] = a[i];
    } 
}

// function to swap elements
__device__ void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}

// function to find the partition position
__device__ int partition(int array[], int low, int high) {
  
  // select the rightmost element as pivot
  int pivot = array[high];
  
  // pointer for greater element
  int i = (low - 1);

  // traverse each element of the array
  // compare them with the pivot
  for (int j = low; j < high; j++) {
    if (array[j] <= pivot) {
        
      // if element smaller than pivot is found
      // swap it with the greater element pointed by i
      i++;
      
      // swap element at i with element at j
      swap(&array[i], &array[j]);
    }
  }

  // swap the pivot element with the greater element at i
  swap(&array[i + 1], &array[high]);
  
  // return the partition point
  return (i + 1);
}

__device__ void quickSort(int array[], int low, int high) {
  if (low < high) {
    
    // find the pivot element such that
    // elements smaller than pivot are on left of pivot
    // elements greater than pivot are on right of pivot
    int pi = partition(array, low, high);
    
    // recursive call on the left of pivot
    quickSort(array, low, pi - 1);
    
    // recursive call on the right of pivot
    quickSort(array, pi + 1, high);
  }
}


__global__ void merge_sort(int *a, int n, int samples){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

}


void merge_par(int *a, int n){
    int *d_a;
    int size = n * sizeof(int);
    int blocks = 1;
    int threads = 1;

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    merge_sort<<<blocks, threads>>>(d_a, n, blocks);
    //sort_in<<<(n + 1023) / 1024, 1024>>>(d_a, n);  
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
}

int main(){

    int *a, *aux;
    const size_t size = N * sizeof(int);

    a = (int *)malloc(size);

    // Init Array
    for(int i=0; i < N; i++){
        a[i] = rand();
        printf("%d ", a[i]);
    }

    printf("\nStarted...\n");
    clock_t start = clock();
    merge_par(a, N);
    clock_t stop = clock();
    printf("Finished sorting in %lf seconds\n", (double)(stop-start) / CLOCKS_PER_SEC);

    for(int i = 0; i < N; i++){
        printf("%d ", a[i]);
    }
        
}