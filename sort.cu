#include <stdio.h>
#include <time.h>

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


int main() {
    
    const int N = 1000;
    clock_t start, stop;
    int a[N], b[N];
    bool good = true;

    init_array(a, N);
    copy_array(a, b, N);

    for(int i=0; i < N; i++){
        printf("%d ", a[i]);
    }
    printf("\n");
    for(int i=0; i < N; i++){
        printf("%d ", b[i]);
    }
    printf("\n");


    selection_sort(a, N);
    start = clock();
    shell_sort(b, N, 2);
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
