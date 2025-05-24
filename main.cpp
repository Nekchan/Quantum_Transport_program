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

arma::cx_mat hamiltonian_1D(int len, arma::cx_mat t)
{
    arma::cx_mat Hamiltonian;
    arma::cx_mat H_I = -2*arma::eye<arma::cx_mat>(len, len);
    arma::cx_mat H_down = arma::zeros<arma::cx_mat>(len, len);
    arma::cx_mat H_up = arma::zeros<arma::cx_mat>(len, len);
    for(size_t i = 0; i<len-1; i++)
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


std::vector<arma::cx_mat> retarted_GF(int len, std::vector<arma::cx_double> Energy,  std::vector<arma::cx_mat> self_energy_left, std::vector<arma::cx_mat> self_energy_right, arma::cx_mat Hamiltonian)
{
    arma::cx_mat mat_2_inv;
    arma::cx_mat tmp;
    arma::cx_mat self_energy_left_mat = arma::zeros<arma::cx_mat>(len, len);
    arma::cx_mat self_energy_right_mat = arma::zeros<arma::cx_mat>(len, len);
    self_energy_left_mat(0,0) = 1;
    self_energy_right_mat(len-1,len-1) = 1;
    std::vector<arma::cx_mat> GF_retarted;
    for(size_t i = 0; i< Energy.size(); i++)
    {
        self_energy_left_mat = arma::kron(self_energy_left_mat, self_energy_left[i]);
        self_energy_right_mat = arma::kron(self_energy_right_mat, self_energy_right[i]);
        mat_2_inv = Energy[i] - Hamiltonian - self_energy_left_mat - self_energy_right_mat;
        tmp = arma::inv(mat_2_inv);
        GF_retarted.push_back(tmp);
    }
    return GF_retarted;
}

std::vector<arma::cx_double> Transport_function(std::vector<arma::cx_mat> retarded_Greens_function, std::vector<arma::cx_mat> gamma_left, std::vector<arma::cx_mat> gamma_right)
{
    std::vector<arma::cx_double> transport_function;
    arma::cx_double G_1N;
    for(size_t i = 0; i<retarded_Greens_function.size(); i++)
    {
        G_1N = retarded_Greens_function[i](0, retarded_Greens_function[i].n_cols-1);
        transport_function.push_back(gamma_left[i](0,0)*gamma_right[i](0,0)*std::norm(G_1N));
    }
    return transport_function;
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
    
    int n = 500;

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
    
    std::vector<arma::cx_mat> GF_retarted = retarted_GF(size, E, self_energy_left, self_energy_right, H_sample);

    std::vector<arma::cx_double> T = Transport_function(GF_retarted, gamma_left, gamma_right);

    double e = 1;
    double h = 4.135667662e-15;

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
        surf_greens_fun << E[i].real() << " " << g_surface[i](0,0).real() << " " << g_surface[i](0,0).imag() << std::endl;
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

    std::fstream Greens_fun_retarded("Greens_fun_retarded.dat", std::ios::out);

    if(!Greens_fun_retarded)
    {
        std::system("touch Greens_fun_retarded.dat");
    }

    if(!Greens_fun_retarded)
    {
        std::cerr << "File Greens_fun_retarded.dat was unable to open";
    }

    for(size_t k = 0; k< GF_retarted.size(); k++)
    {
        Greens_fun_retarded << "Retarded Green's function for E=" << E[k] << std::endl;
        for(size_t i = 0; i<size; i++)
        {
            for(size_t j = 0; j<size; j++)
            {
                Greens_fun_retarded << GF_retarted[k](i, j) << " ";
            }
            Greens_fun_retarded << std::endl;
        }
    }
    

    Greens_fun_retarded.close();
    std::cout << "Retarded Greens function saved to a file succesfully\n";

    std::fstream transp_fun("transp_fun.dat", std::ios::out);

    if(!transp_fun)
    {
        std::system("touch transp_fun.dat");
    }

    if(!transp_fun)
    {
        std::cerr << "File transp_fun.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        transp_fun << E[i].real() << " " << T[i].real() << " " << T[i].imag() << std::endl;
    }
    
    transp_fun.close();
    std::cout << "Transport function saved to a file succesfully\n";

    std::fstream conductance("conductance.dat", std::ios::out);

    if(!conductance)
    {
        std::system("touch conductance.dat");
    }

    if(!conductance)
    {
        std::cerr << "File conductance.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        conductance << E[i].real() << " " << (e*e/h)*T[i].real() << " " << (e*e/h)*T[i].imag() << std::endl;
    }
    
    conductance.close();
    std::cout << "Conductance saved to a file succesfully\n";

    return 0;
}