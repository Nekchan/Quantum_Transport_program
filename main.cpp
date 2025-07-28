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
    for(int i = 0; i<N; i++)
    {
        Ei = arma::cx_double(E_min + i*dE, eta);
        Energy.push_back(Ei);
    }
    return Energy;
}

arma::cx_mat hamiltonian(size_t xsize, arma::cx_mat h, arma::cx_mat u)
{
    arma::cx_mat Hamiltonian;
    arma::cx_mat H_I = arma::eye<arma::cx_mat>(xsize, xsize);
    arma::cx_mat H_down = arma::zeros<arma::cx_mat>(xsize, xsize);
    arma::cx_mat H_up = arma::zeros<arma::cx_mat>(xsize, xsize);
    for(size_t i = 0; i<xsize-1; i++)
    {
        H_down(i+1,i) = 1;
        H_up(i,i+1) = 1;
    }
    Hamiltonian = arma::kron(H_I, h) + arma::kron(H_down, u) + arma::kron(H_up, u);
    return Hamiltonian;
}

arma::cx_mat block_matrix(size_t len, arma::cx_double node, arma::cx_double t)
{
    arma::cx_mat block;
    arma::cx_mat H_I = arma::eye<arma::cx_mat>(len, len);
    arma::cx_mat H_down = arma::zeros<arma::cx_mat>(len, len);
    arma::cx_mat H_up = arma::zeros<arma::cx_mat>(len, len);
    for(size_t i = 0; i<len-1; i++)
    {
        H_down(i+1,i) = -1;
        H_up(i,i+1) = -1;
    }
    block = node*H_I + t*H_down + t*H_up;
    return block;
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

std::vector<arma::cx_mat> sancho_rubio_algorithm(std::vector<arma::cx_double> Energy, arma::cx_mat h, arma::cx_mat u, double tol = 1e-8, long int max_iter = 1000000)
{
    arma::cx_mat g_surface_i;
    arma::cx_mat alpha;
    arma::cx_mat beta;
    arma::cx_mat epsilon;
    arma::cx_mat epsilon_s;
    arma::cx_mat epsilon_s_prev;
    arma::cx_mat tmp;
    std::vector<arma::cx_mat> g_surface;
    double delta;
    int iter;
    for(size_t i =0; i< Energy.size(); i++)
    {
        tmp = arma::solve(Energy[i]*arma::eye<arma::cx_mat>(u.n_rows, u.n_cols)-h, arma::eye<arma::cx_mat>(h.n_rows, h.n_cols));
        alpha = u*tmp*u;
        beta = arma::trans(u)*tmp*arma::trans(u);
        epsilon_s = h + u*tmp*arma::trans(u);
        epsilon = epsilon_s + arma::trans(u)*tmp*u;
        

        delta = 1.0;
        iter = 0;

        while(delta>=tol && iter<=max_iter)
        {
            tmp = arma::solve(Energy[i]*arma::eye<arma::cx_mat>(u.n_rows, u.n_cols)-epsilon, arma::eye<arma::cx_mat>(epsilon.n_rows, epsilon.n_cols));
            epsilon_s_prev = epsilon_s;
            epsilon_s += alpha*tmp*beta;
            epsilon +=beta*tmp*alpha +alpha*tmp*beta;
            alpha = alpha*tmp*alpha;
            beta = beta*tmp*beta;
            
            delta = arma::norm(epsilon_s - epsilon_s_prev, "fro");
            iter++;
        }
        if (delta > 1e-3) std::cout << "Did not converge for E=" << Energy[i].real() << "\n";
        if (iter == max_iter) std::cout << "Did not converge for E=" << Energy[i].real() << "\n";

        g_surface_i = arma::solve(Energy[i]*arma::eye<arma::cx_mat>(u.n_rows, u.n_cols)-epsilon_s, arma::eye<arma::cx_mat>(epsilon_s.n_rows, epsilon_s.n_cols));
        g_surface.push_back(g_surface_i);
    }
    return g_surface;
}


std::vector<arma::cx_mat> retarted_GF(size_t xsize, std::vector<arma::cx_double> Energy,  std::vector<arma::cx_mat> self_energy_left, std::vector<arma::cx_mat> self_energy_right, arma::cx_mat Hamiltonian)
{
    arma::cx_mat mat_2_inv;
    arma::cx_mat tmp;
    std::vector<arma::cx_mat> GF_retarted;
    arma::cx_mat H;
    arma::cx_mat self_energies_left;
    arma::cx_mat self_energies_right;
    
    for(size_t i = 0; i< Energy.size(); i++)
    {
        self_energies_left = arma::zeros<arma::cx_mat>(xsize, xsize);
        self_energies_left(0,0) = 1;
        self_energies_left = arma::kron(self_energies_left, self_energy_left[i]);

        self_energies_right = arma::zeros<arma::cx_mat>(xsize, xsize);
        self_energies_right(xsize-1, xsize-1) = 1;
        self_energies_right = arma::kron(self_energies_right, self_energy_right[i]);

        H=Hamiltonian+self_energies_left+self_energies_right;

        mat_2_inv = Energy[i]*arma::eye<arma::cx_mat>(H.n_rows, H.n_cols) - H;
        tmp = arma::solve(mat_2_inv, arma::eye<arma::cx_mat>(mat_2_inv.n_rows, mat_2_inv.n_cols), arma::solve_opts::likely_sympd);
        GF_retarted.push_back(tmp);
    }
    return GF_retarted;
}

std::vector<arma::cx_double> Transmission_function(size_t xsize, std::vector<arma::cx_mat> retarded_Greens_function, std::vector<arma::cx_mat> gamma_left, std::vector<arma::cx_mat> gamma_right)
{
    std::vector<arma::cx_double> transmission_function;
    arma::cx_mat GF_advanced;
    for(size_t i = 0; i<retarded_Greens_function.size(); i++)
    {
        GF_advanced = arma::trans(retarded_Greens_function[i]);
        transmission_function.push_back(arma::trace(GF_advanced.submat(0, 0, xsize-1, xsize-1)*gamma_right[i]*retarded_Greens_function[i].submat(0, 0, xsize-1, xsize-1)*gamma_left[i]));
        
    }
    return transmission_function;
}

void ldos(std::vector<arma::cx_mat>& spectral_function, std::vector<arma::cx_double>& density_of_states, std::vector<arma::cx_mat> retarted_Greens_functions)
{
    arma::cx_mat advanced_Greens_functions;
    arma::cx_double ii = arma::cx_double(0,1);
    for(size_t i = 0; i < retarted_Greens_functions.size(); i++)
    {
        advanced_Greens_functions = arma::trans(retarted_Greens_functions[i]);
        spectral_function.push_back(ii*(retarted_Greens_functions[i]-advanced_Greens_functions));
        density_of_states.push_back(arma::trace(spectral_function[i])/(2*M_PI));
    }
}




int main()
{
    
    //initial values required for the energy
    double E_min = -3;            //minimum energy in eV
    const double eta = 1e-18;       //a small constant approaching zero that models discontinuity in eV
    double dE = 0.001;               //the difference between next and previous energy eV    
    int N = 9/dE;                   //number of points calculated
    size_t ysize = 3;
    size_t xsize = 3;


    //Assaining an energy
    std::vector<arma::cx_double> E = Energy(N, E_min, dE, eta);     //energy data of the system in eV
    
    //calculating the aproximate surface Green's function
    arma::cx_double t = arma::cx_double(1, 0);    //matrix containing hopping integrals
    arma::cx_mat h = block_matrix(ysize, arma::cx_double(2,0)*t, -t);
    arma::cx_mat u = block_matrix(ysize, -t, 0);


    std::vector<arma::cx_mat> g_surface = sancho_rubio_algorithm(E, h, u);

    arma::cx_double ii = arma::cx_double(0,1);

    
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

    arma::cx_mat H_sample = hamiltonian(xsize, h, u);
    
    std::vector<arma::cx_mat> GF_retarted = retarted_GF(xsize, E, self_energy_left, self_energy_right, H_sample);

    std::vector<arma::cx_mat> GF_advanced;
    
    for(size_t i =0; i<GF_retarted.size(); i++)
    {
        GF_advanced.push_back(arma::trans(GF_retarted[i]));
    }

    std::vector<arma::cx_double> T = Transmission_function(xsize, GF_retarted, gamma_left, gamma_right);

    double e = 1;
    double h_planck = 4.135667662e-15;

    std::vector<arma::cx_mat> A;

    std::vector<arma::cx_double> DOS;

    ldos(A, DOS, GF_retarted);

    std::cout<<"Calculations ended succesfully\n";

    //Saving to file
    if(false)
    {
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

        for(size_t i = 0; i<xsize; i++)
        {
            for(size_t j = 0; j<xsize; j++)
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
            Greens_fun_retarded << E[k].real() << " " << GF_retarted[k](0, GF_retarted[k].n_cols-1).real() << " " << GF_retarted[k](0, GF_retarted[k].n_cols-1).imag() << std::endl;
        }
    
    

        Greens_fun_retarded.close();
        std::cout << "Retarded Greens function saved to a file succesfully\n";
    }
    

    std::fstream transm_fun("transm_fun.dat", std::ios::out);

    if(!transm_fun)
    {
        std::system("touch transm_fun.dat");
    }

    if(!transm_fun)
    {
        std::cerr << "File transm_fun.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        transm_fun << E[i].real() << " " << T[i].real() << " " << T[i].imag() << std::endl;
    }
    
    transm_fun.close();
    std::cout << "Transmission function saved to a file succesfully\n";

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
        conductance << E[i].real() << " " << (e*e/h_planck)*T[i].real() << " " << (e*e/h_planck)*T[i].imag() << std::endl;
    }
    
    conductance.close();
    std::cout << "Conductance saved to a file succesfully\n";

    std::fstream LDOS("LDOS.dat", std::ios::out);

    if(!LDOS)
    {
        std::system("touch LDOS.dat");
    }

    if(!LDOS)
    {
        std::cerr << "File LDOS.dat was unable to open";
        return 1;
    }

    
    for(size_t i = 0; i<E.size(); i++)
    {
        LDOS << E[i].real() << " " << DOS[i].real() << " " << DOS[i].imag() << std::endl;
    }
    
    LDOS.close();
    std::cout << "Local density of states saved to a file succesfully\n";


    return 0;
}