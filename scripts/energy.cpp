#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> energy_calc_msa(
           py::array_t<long> msa, /* shape (num_seqs, L) */
           py::array_t<double> h_i_a,  /* shape (L, q) */
           py::array_t<double> e_i_a_j_b) /* shape (L, q, L, q) */ {

    auto buf_m = msa.unchecked<2>(); 
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    if (buf_h.ndim() != 2) 
        throw std::runtime_error("Number of dimensions for fields must be 2");
    if (buf_e.ndim() != 4) 
        throw std::runtime_error("Number of dimensions for couplings must be 4");

    if (buf_m.shape(1) != buf_h.shape(0))
        throw std::runtime_error("msa and fields do not match "
                                 "along the length dimension");
    if (buf_h.shape(0) != buf_e.shape(0))
        throw std::runtime_error("couplings and fields do not match "
                                 "along the length dimension");
    if (buf_e.shape(0) != buf_e.shape(2))
        throw std::runtime_error("couplings do not match "
                                 "along the length dimension");

    if (buf_h.shape(1) != buf_e.shape(1))
        throw std::runtime_error("couplings and fields do not match "
                                 "along the alphabet dimension");
    if (buf_e.shape(1) != buf_e.shape(3))
        throw std::runtime_error("couplings do not match "
                                 "along the alphabet dimension");

    size_t num_seqs = (unsigned) buf_m.shape(0);
    size_t L = (unsigned) buf_h.shape(0);
    //size_t q = (unsigned) buf_h.shape(1);

    // Resulting Energy Array
    auto result = py::array_t<double>(num_seqs);
    auto buf_r = result.request();

    double  *ptr_r = (double *) buf_r.ptr;


    for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        //loop over each sequence    
        double energy = 0.;
        for (size_t i = 0; i < L ; i++) {
            // loop over each residue position
            auto i_idx = buf_m(seq_idx, i); // AA index
            energy += buf_h(i, i_idx );
            for (size_t j = i+1; j < L; j++) {
                // double loop over residue position
                auto j_idx = buf_m(seq_idx, j); // AA index
                energy += buf_e(i, i_idx, j, j_idx);
            }
        }
        ptr_r[seq_idx] = -energy;
    }

    return result;
}


void check_canonical_parameter_shapes(py::array_t<double> h_i_a,
                                      py::array_t<double> e_i_a_j_b) {
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    if (buf_h.ndim() != 2) 
        throw std::runtime_error("Number of dimensions for fields must be 2");
    if (buf_e.ndim() != 4) 
        throw std::runtime_error("Number of dimensions for couplings must be 4");

    if (buf_h.shape(0) != buf_e.shape(0))
        throw std::runtime_error("couplings and fields do not match "
                                 "along the length dimension");
    if (buf_e.shape(0) != buf_e.shape(2))
        throw std::runtime_error("couplings do not match "
                                 "along the length dimension");

    if (buf_h.shape(1) != buf_e.shape(1))
        throw std::runtime_error("couplings and fields do not match "
                                 "along the alphabet dimension");
    if (buf_e.shape(1) != buf_e.shape(3))
        throw std::runtime_error("couplings do not match "
                                 "along the alphabet dimension");

}



py::tuple energy_calc_single_mutants(
                py::array_t<long> wt, /* shape (L, ) */
                py::array_t<double> h_i_a,  /* shape (L, q) */
                py::array_t<double> e_i_a_j_b )  {

    check_canonical_parameter_shapes(h_i_a, e_i_a_j_b);

    auto buf_w = wt.unchecked<1>(); 
    auto buf_h = h_i_a.unchecked<2>(); 
    auto buf_e = e_i_a_j_b.unchecked<4>();

    if (buf_w.shape(0) != buf_h.shape(0))
        throw std::runtime_error("sequence and fields do not match "
                                 "along the length dimension");

    size_t L = (unsigned) buf_h.shape(0);
    size_t q = (unsigned) buf_h.shape(1);

    // Resulting Energy Array
    size_t num_mutants = L * (q-1);
    auto mut_shape = pybind11::array::ShapeContainer({(long) num_mutants, 2});
    auto mutants = py::array_t<long>( mut_shape);
    auto buf_m = mutants.request();
    auto writer_m = mutants.mutable_unchecked<2>();

    auto energies = py::array_t<double>(num_mutants);
    auto buf_en = energies.request();
    auto writer_en = energies.mutable_unchecked<1>();

    // Calculate the energy of the sequence passed in
    double energy_wt = 0;
    for (size_t i = 0; i < L ; i++) {
        auto aa_i = buf_w(i); // AA index of sequence
        energy_wt += buf_h(i, aa_i);
        for (size_t j = i+1; j < L; j++) {
            // double loop over residue position
            auto aa_j = buf_w(j); // AA index
            energy_wt += buf_e(i, aa_i, j, aa_j);
        }
    }
    
    // Now calculate the energy of the single mutants
    size_t mutant_counter = 0;
    for (size_t i = 0 ; i < L; i++) {
        // loop over each residue position
        auto wt_i = (unsigned) buf_w(i); // AA index of sequence
        double energy_wt_i = buf_h(i, wt_i);
        for (size_t j = 0; j < L; j++) {
            if (j == i) continue;
            if (j < i) {
                energy_wt_i += buf_e(j, buf_w(j), i, wt_i);
            } else {
                energy_wt_i += buf_e(i, wt_i, j, buf_w(j));
            }
        }
        for (size_t a = 0; a < q; a++) {
            if (a == wt_i) continue; // this is wt and not a mutant
            auto energy_mut_i = buf_h(i, a);
            for (size_t j = 0; j < L; j++) {
                if (j == i) continue;
                if (j < i) {
                    energy_mut_i += buf_e(j, buf_w(j), i, a);
                } else {
                    energy_mut_i += buf_e(i, a, j, buf_w(j));
                }
            }
            if (mutant_counter >= num_mutants) {
                throw std::runtime_error("mutant counter larger than"
                                " num_mutants"); 
            }
            writer_m(mutant_counter, 0) = i;
            writer_m(mutant_counter, 1) = a;
            auto energy_mut = energy_mut_i - energy_wt_i + energy_wt; 
            writer_en(mutant_counter) = -energy_mut;

            ++mutant_counter;
        }
    }

    return py::make_tuple(mutants, energies);
}

PYBIND11_MODULE(energy, m) {
    m.doc() = "pybind11 Energy calculator"; // optional module docstring

    m.def("energy_calc_msa", &energy_calc_msa, 
                    "Compute the energy of each sequence in an MSA");
    m.def("energy_calc_single_mutants", &energy_calc_single_mutants, 
                    "Compute the energy of single mutants of a sequence");
}
