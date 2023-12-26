#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

int main()
{
    const int N = 1024;
    double *phi = new double [N*N];
    double *phin = new double [N*N];
    
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            phi[i*N+j] = 1.0;
            phin[i*N+j] = 1.0;
        }
    }

    auto start = system_clock::now();
    for (int n=0; n<1000; ++n){
        for (int i=1; i<N-1; ++i) {
            for (int j=1; j<N-1; ++j) {
                phin[i*N+j] = 0.25*(phi[(i-1)*N+j]+ \
                                    phi[(i+1)*N+j]+ \
                                    phi[i*N+j-1]+ \
                                    phi[i*N+j+1]);
            }
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end-start);
    cout << "takes "<< double(duration.count())*microseconds::period::num/microseconds::period::den << "s" << endl;

    delete [] phi;
    delete [] phin;
    return 0;
}