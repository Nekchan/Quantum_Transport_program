#include<iostream>
#include<armadillo>
#include<vector>
#include<complex>

/*
-------------Energy function---------------
This function returns a vector with energy values expressed in complex numbers.
The vector can later be used for further calculations.
The function needs the following input values:
    -int N  - number of values,
    -double E_min - minimal energy (can be negative),
    -double dE - the difference between different neighbouring values of energy,
    -double eta - a small complex value that models discontinuity (it should approach zero), 
*/

std::vector<std::complex<double>> Energy(int N, double E_min, double dE, const double eta)
{
    std::vector<std::complex<double>> Energy;
    std::complex<double> Ei = 0;
    for(int i = 0; i<=N; i++)
    {
        Ei = E_min + i*dE +1j*eta;
        Energy.push_back(Ei);
    }
    return Energy;
}


int main()
{
    double E_min = -1.0;            //minimum energy in eV
    const double eta = 1e-24;       //a small constant approaching zero that models discontinuity in eV
    double dE = 0.00001;            //the difference between next and previous energy eV    
    int N = 6/dE;

    std::vector<std::complex<double>> E = Energy(N, E_min, dE, eta);
    for(int i = 0; i< E.size(); i++)
    {
        std::cout<< E[i]<<std::endl;
    }

    return 0;
}