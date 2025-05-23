#include<iostream>
#include<armadillo>
#include<vector>
#include<complex>
#include<fstream>


/*
-------------Energy function---------------
This function returns a vector<cx_double> with energy values expressed in complex numbers.
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

arma::cx_mat hamiltonian_1D(int size, arma::cx_mat t)
{
    arma::cx_mat Hamiltonian;
    arma::cx_mat H_I = -2*arma::eye<arma::cx_mat>(size, size);
    arma::cx_mat H_down = arma::zeros<arma::cx_mat>(size, size);
    arma::cx_mat H_up = arma::zeros<arma::cx_mat>(size, size);
    for(size_t i = 0; i<size-1; i++)
    {
        H_down(i+1,i) = 1;
        H_up(i,i+1) = 1;
    }
    Hamiltonian = arma::kron(H_I, t) + arma::kron(H_down, t) + arma::kron(H_up, t);
    return Hamiltonian;
}


/*------------Sancho-Rubio algorithm--------------
This function calculates an aproximapate surface Green's function returned as a vector<cx_mat>. I designed it, so that 
(hopefully) it can be easilly scaled up to a 2D system. 
The function needs following inputs:
    -int n - number of iteration through which the algorithm has to go through (hopefully in 
    the future I will add a convergence criterium, so that this number won't be necessary),
    -vector<cx_double> Energy - vector which contains the energy values of the system,
    -cx_mat t - a matrix that posesses the hopping integrals of the system.
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
    
    //initial values required for the energy
    double E_min = -1.0;            //minimum energy in eV
    const double eta = 1e-18;       //a small constant approaching zero that models discontinuity in eV
    double dE = 0.001;               //the difference between next and previous energy eV    
    int N = 6/dE;                   //number of points calculated
    int size = 10;

    //Assaining an energy
    std::vector<arma::cx_double> E = Energy(N, E_min, dE, eta);     //energy data of the system in eV
    
    //calculating the aproximate surface Green's function
    arma::cx_mat t = arma::ones<arma::cx_mat>(1, 1);    //matrix containing hopping integrals
    t(0, 0) = 1;
    
    int n = 100;

    std::vector<arma::cx_mat> g_surface = sancho_rubio_algorithm(n, E, t);

    arma::cx_double ii = arma::cx_double(0,1);

    arma::cx_mat u = -t;
    std::vector<arma::cx_mat> self_energy_left;
    std::vector<arma::cx_mat> self_energy_right;
    std::vector<arma::cx_mat> gamma_left;
    std::vector<arma::cx_mat> gamma_right;
    for(size_t i =0; i< E.size(); i++)
    {
        self_energy_left.push_back(u*g_surface[i]*arma::trans(u));
        self_energy_right.push_back(arma::trans(u)*g_surface[i]*u);
        gamma_left.push_back(ii*(self_energy_left[i] - arma::trans(self_energy_left[i])));
        gamma_right.push_back(ii*(self_energy_right[i] - arma::trans(self_energy_right[i])));
    }

    arma::cx_mat H_sample = hamiltonian_1D(size, t);
    
    

    std::cout<<"Calculations ended succesfully\n";

    //Saving to file
    std::fstream surf_greens_fun("surf_greens_fun.dat", std::ios::out);

    if(!surf_greens_fun)
    {
        std::system("touch surf_greens_fun.dat");
    }

    if(!surf_greens_fun)
    {
        std::cerr << "File surf_greens_fun.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        surf_greens_fun << E[i].real() << " " << g_surface[i](0,0).real() << " " << g_surface[i](0,0).imag() << " " << self_energy_left[i](0,0).real()  << " " << self_energy_left[i](0,0).imag() << std::endl;
    }
    
    surf_greens_fun.close();
    std::cout << "Surface Greens function saved to a file succesfully\n";

    std::fstream self_energies("self_energies.dat", std::ios::out);

    if(!self_energies)
    {
        std::system("touch self_energies.dat");
    }

    if(!self_energies)
    {
        std::cerr << "File self_energies.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        self_energies << E[i].real() << " " << self_energy_left[i](0,0).real()  << " " << self_energy_left[i](0,0).imag() << " " << self_energy_right[i](0,0).real()  << " " << self_energy_right[i](0,0).imag() << std::endl;
    }
    
    self_energies.close();
    std::cout << "Self energies saved to a file succesfully\n";

    std::fstream gamma("gamma.dat", std::ios::out);

    if(!gamma)
    {
        std::system("touch gamma.dat");
    }

    if(!gamma)
    {
        std::cerr << "File gamma.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        gamma << E[i].real() << " " << gamma_left[i](0,0).real()  << " " << gamma_left[i](0,0).imag() << " " << gamma_right[i](0,0).real()  << " " << gamma_right[i](0,0).imag() << std::endl;
    }
    
    gamma.close();
    std::cout << "Gamma matrices saved to a file succesfully\n";

    std::fstream hamiltonian_sample("hamiltonian_sample.dat", std::ios::out);

    if(!hamiltonian_sample)
    {
        std::system("touch hamiltonian_sample.dat");
    }

    if(!hamiltonian_sample)
    {
        std::cerr << "File hamiltonian_sample.dat was unable to open";
    }

    for(size_t i = 0; i<size; i++)
    {
        for(size_t j = 0; j<size; j++)
        {
            hamiltonian_sample << H_sample(i, j) << " ";
        }
        hamiltonian_sample << std::endl;
    }

    hamiltonian_sample.close();
    std::cout << "Hamiltonian of the sample saved to a file succesfully\n";

    return 0;
}