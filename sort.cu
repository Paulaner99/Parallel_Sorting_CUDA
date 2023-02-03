#include <stdio.h>
#include <time.h>

#define N 20

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void init_array(int *a, int n){
    for(int i=0; i < n; i++){
        a[i] = rand();
        //printf("%d ", a[i]);
    }
    //printf("\n");
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
    /*
    for(int i=0; i < n; i++){
        printf("%d ", a[i]);
    }
    printf("\n");
    */
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
                for(int j=i-gap; j-gap > 0 && a[j-gap] > a[j]; j=j-gap){
                    tmp = a[j];
                    a[j] = a[j-gap];
                    a[j-gap] = tmp;
                }
            }
        }
        clock_t stop = clock();
        printf("GAP %d time: %lf\n", gap, (double)(stop-start) / CLOCKS_PER_SEC);
    }
    /*
    for(int i=0; i < n; i++){
        printf("%d ", a[i]);
    }
    printf("\n");
    */
}

__global__ void sort(int *a, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < n){
        for(int step=0; step < blockDim.x; step++){
            if(step % 2 == 0 && idx % 2 == 0 && threadIdx.x < blockDim.x - 1){
                if(a[idx] > a[idx+1]){
                    int tmp = a[idx+1];
                    a[idx+1] = a[idx];
                    a[idx] = tmp;
                }
            }
            if(step % 2 == 1 && idx % 2 == 1 && threadIdx.x < blockDim.x - 1){
                if(a[idx] > a[idx+1]){
                    int tmp = a[idx+1];
                    a[idx+1] = a[idx];
                    a[idx] = tmp;
                }
            }
            __syncthreads(); 
        }   
    }


}


int main(){

    int *a, *seq, *par;
    int *d_a, *d_seq, *d_par;
    const size_t size = N * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_seq, size);
    cudaMalloc((void **)&d_par, size);
    
    /* Alloc space for host copies of a,b,c and setup input values */
    a = (int *)malloc(size);
    seq = (int *)malloc(size);
    par = (int *)malloc(size);

    // Init Array
    for(int i=0; i < N; i++){
        a[i] = rand() % 100;
        printf("%d ", a[i]);
    }
    printf("\n\n");

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    /* Launch sort() kernel on GPU */
    sort<<<2,N/2>>>(d_a, N);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++){
        printf("%d ", a[i]);
    }

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