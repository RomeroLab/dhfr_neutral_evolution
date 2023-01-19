// Count the number of each type of residue pairs in an MSA
//
// It easier to compute pair counts using sequences instead of using one-hot
// arrays. Computing this way requires jagged arrays which are slow in python.
// So we use this small c++ snippet which we call from python.
//
// `MSA` is a numpy array (uint8)  of shape (N, L) with each element
// `q` is the maximum index of the alphabet
// Return type is a numpy array of ints of shape (L,q,L,q) (i,a,j,b)
//  and each element in the return counts the number of elements 
//  in the MSA that have alphabet a in residue i and alphabet b in residue j.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Count the number of intersections in two containers
template<typename T>
struct SetIntersectionCounter {
    size_t count = 0;
    typedef T value_type;
    void push_back(const T&) { ++count; }
    void clear() { count = 0;}
};

py::array_t<int32_t> calc_paircounts(
                py::array_t<uint8_t> msa, 
                // `msa` : 2D numpy array (np.uint8) representing an MSA
                //          where each element is AA or Codon index 
                unsigned int q /* Alphabet size */ ) {

    auto buf_m = msa.unchecked<2>(); 

    if (buf_m.ndim() != 2) 
        throw std::runtime_error("Number of dimensions for MSA must be 2");
    
    // Check that the maximum index in the msa is less than the alphabet size
    unsigned int max_code = msa.attr("max")().cast<unsigned int>();
    if (max_code >= q) {
        throw std::runtime_error("Max element of MSA must be" 
                                 " less than alphabet size");
    }

    py::size_t num_seqs = (unsigned) buf_m.shape(0);
    py::size_t L = (unsigned) buf_m.shape(1);

    auto qs = (size_t) q; // convert to size_t
   
    // For each pair (i, a) store the sequence number in Lq_indices
    // if that sequence has alphabet a in position i
    typedef std::vector< std::vector<int> > Arr2D;
    typedef std::vector< Arr2D > Arr3D;
    // container for sequence indices
    Arr3D Lq_indices(L, Arr2D(q, std::vector<int>()) );
    for (size_t i = 0; i < L ; i++) {
        auto &vec_ptr_i = Lq_indices[i];
        //loop over each sequence    
        for (size_t seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
            uint8_t a = buf_m(seq_idx, i);
            // save the sequence index in the right container
            vec_ptr_i[a].push_back(seq_idx);
        }
    }
    
    // Result : paircount Array
    auto result = py::array_t<int>({L, qs, L, qs});
    auto result_info = result.request();
    auto buf_r = result.mutable_unchecked<4>();
 
    // Now we count the intersections between the containers
    // As the containers are sorted we can use std::set_intersection
    SetIntersectionCounter<int> counter; 
    for (size_t i = 0; i < L; i++) {
        auto &vec_ptr_i = Lq_indices[i];
        for(size_t j = i; j < L; j++) { 
            // j=i case could be done separately but may not save much time
            auto &vec_ptr_j = Lq_indices[j]; 
            for (size_t a = 0; a < qs; a++) {
                auto &vec_ptr_i_a = vec_ptr_i[a];
                if (vec_ptr_i_a.size()) { // We have counts at (i,a)
                    for (size_t b = 0; b < qs; b++) {
                        counter.clear();
                        std::set_intersection(
                                vec_ptr_i_a.begin(), vec_ptr_i_a.end(),
                                vec_ptr_j[b].begin(), vec_ptr_j[b].end(),
                                std::back_inserter(counter));
                        buf_r(i,a,j,b) = counter.count;
                        buf_r(j,b,i,a) = counter.count; // symmetry
                    } 
                } else { // We have no counts at (i,a)
                    // This case is quite likely as we have a sparse MSA
                    for (size_t b = 0; b < qs; b++) {
                        buf_r(i,a,j,b) = 0;
                        buf_r(j,b,i,a) = 0; // symmetry
                    } // end for loop
                } // end if-else
            } // end a-loop
        } // end j-loop
    } // end i-loop
    return result;
}

PYBIND11_MODULE(paircounts, m) {
    m.doc() = "pybind11 Paircount calculator"; // optional module docstring

    m.def("calc_paircounts", &calc_paircounts, 
               "Compute the paircounts at each pair of residues for an MSA",
               py::arg("msa"), py::arg("q"));
}
