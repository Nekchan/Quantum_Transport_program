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

std::vector<arma::cx_double> Energy(int N, double E_min, double dE, const double eta)
{
    std::vector<arma::cx_double> Energy;
    arma::cx_double Ei = arma::cx_double(0, 0);
    for(int i = 0; i<=N; i++)
    {
        Ei = arma::cx_double(E_min + i*dE, eta);
        Energy.push_back(Ei);
    }
    return Energy;
}

/*------------Sancho-Rubio algorithm--------------

*/

std::vector<arma::cx_mat> sancho_rubio_algorithm(int n, std::vector<arma::cx_double> Energy, arma::cx_mat t)
{
    arma::cx_mat A = -t;
    arma::cx_mat B = -t;
    arma::cx_mat g_surface_i;
    arma::cx_mat D;
    arma::cx_mat alpha;
    arma::cx_mat beta;
    arma::cx_mat epsilon;
    arma::cx_mat epsilon_s;
    arma::cx_mat tmp;
    std::vector<arma::cx_mat> g_surface;
    std::cout << "XD";
    for(size_t i =0; i< Energy.size(); i++)
    {
        D = Energy[i]*arma::eye<arma::cx_mat>(t.n_rows, t.n_cols) -2*t;
        tmp = arma::inv(D);
        alpha = A*tmp*A;
        beta = B*tmp*B;
        epsilon = D - B*tmp*A - A*tmp*B;
        epsilon_s = D - A*tmp*B;
        for(int k = 0; k<=n; k++)
        {
            tmp = arma::inv(epsilon);
            epsilon -=beta*tmp*alpha +alpha*tmp*beta;
            epsilon_s -= alpha*tmp*beta;
            alpha = alpha*tmp*alpha;
            beta = beta*tmp*beta;
        }
        g_surface_i = arma::inv(epsilon_s);
        g_surface.push_back(g_surface_i);
    }
    return g_surface;
}


int main()
{
    double E_min = -1.0;            //minimum energy in eV
    const double eta = 1e-12;       //a small constant approaching zero that models discontinuity in eV
    double dE = 0.01;            //the difference between next and previous energy eV    
    int N = 6/dE;

    std::vector<arma::cx_double> E = Energy(N, E_min, dE, eta);
    
    arma::cx_mat t = arma::ones<arma::cx_mat>(1, 1);
    t(0, 0) = 1;
    
    int n = 100;

    std::vector<arma::cx_mat> g_surface = sancho_rubio_algorithm(n, E, t);

    for(size_t i = 0; i<E.size(); i++)
    {
        std::cout << E[i] << " " << g_surface[i] << std::endl;
    }


    return 0;
}