"""Model data using codons and find optimal energy parameters """

import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn

import utils
import config


def calc_count_data(msa_filenames_dict=config.MSA_FILESd,
        max_seq_length=config.L):
    """ Calculate and load 1st and 2nd order frequency counts
        This function sorts the msa_filenames_dict by key (round_num)
    """
    counts_main_d = {} # counts of each main effect for each round
    counts_int_d = {} # counts of each interaction effect for each round
    for round_num in sorted(msa_filenames_dict):
        filename = msa_filenames_dict[round_num]
        logging.info(f"Calculating frequency counts for {filename}")
        counts_main_d[round_num], counts_int_d[round_num] =  \
              utils.calc_mutant_counts(filename, max_seq_length=max_seq_length)
    return counts_main_d, counts_int_d

def calc_num_seqsd(counts_main_d):
    """ Sum up counts in 1st order effects to find number of sequences """
    return {k:v[0, :].sum().item() for k,v in counts_main_d.items()}

def calc_l2(arr):
    """ Calculate square of l2 norm"""
    return torch.mul(arr, arr).sum()

def marginalize_int_probs(int_probs):
    L = int_probs.shape[0]
    return (zero_diagonal_int_params(int_probs) 
                            / (L-1)).sum(axis=(2,3))

def calc_margin_loss(main_params, int_params):
    """ Here we marginalize the second order frequencies and subtract
        the first order frequences. Then we calculate the square of the
        L2 norm of this difference and normalize by the length of
        the protein and the size of the alphabet.
        `main_params` : (i, a) float array of shape (L, q)
                                (exp sums to 1 for each value of i)
        if main_params is None then it will return zero
        `int_params` : (i, a, j, b) float array of shape (L, q, L, q)
                                (exp sums to 1 for each value of i, j)
    """
    ret = torch.tensor(0.)
    if main_params is not None:
        first_order_probs = main_params.exp()
        second_order_probs = int_params.exp()
        # marginalize the second order frequencies.
        margin_tensor =  marginalize_int_probs(second_order_probs)
        ret = calc_l2(margin_tensor - first_order_probs)
    return ret

def symmetrize_int_params(int_params):
    return (int_params + int_params.permute(2,3,0,1)) / 2.

def zero_diagonal_int_params(int_params):
    """ Zero out the i=j terms in the array (L,q,L,q)"""
    L = int_params.shape[0]
    return int_params * (1 - torch.eye(L)).reshape(L, 1, L, 1)

def dist_from_symmetry_int_params(x):
    return torch.norm(x - x.permute(2,3,0,1))

def dist_from_symmetry_2d(x):
    return torch.norm(x - x.permute(1, 0))

def convert_main_fitness_to_probs(main_fitness):
    return main_fitness / main_fitness.sum(1, keepdim=True)

def convert_int_fitness_to_probs(int_fitness):
    return int_fitness / int_fitness.sum((1,3), keepdim=True)

def calc_transition_matrices(nt_trans_mat_csv, dtype=config.PRECISION):
    """ Calculate the main effects and interaction effects transition
        matrices
    """
    ## Transition Matrix 61x61
    codon_ex_stops_trans_mat = utils.create_codon_trans_mat_from_csv(
                                        nt_trans_mat_csv)

    # Prepare main effects transition matrix
    # t: `to` residue
    # f: `from` residue
    # Transition matrix trans_mat[t,f] = p(f -> t)
    trans_mat = torch.DoubleTensor(codon_ex_stops_trans_mat).to(
                                                    dtype=dtype)

    # Prepare interaction effects transition matrix
    # t: `to` residue
    # f: `from` residue
    # o: `to` residue
    # r: `from` residue
    # p((t,o) -> (f, r)) = double_trans_mat[t,f,o,r]
    #                    = trans_mat[t,f] * trans_mat[o, r]
    double_trans_mat = torch.einsum("tf,or->tfor", trans_mat, trans_mat)

    return trans_mat, double_trans_mat



class EvolutionCodonModel(torch.nn.Module):

    """ Base class to infer DCA on the codons. This mostly handles the data and
        sets up tensors. The parameters should be in classes derived from this
        one.  
    """

    qc = config.qc
    PRECISION = config.PRECISION
    EPSILON = config.EPSILON
    CODON_AA_MATRIX = config.CODON_AA_MATRIX
    CODON_AA_MAP = config.CODON_AA_MAP

    def __init__(self, L=config.L, 
                 model_name = "DHFR",
                 msa_filenames_dict=config.MSA_FILESd,
                 nt_trans_mat_csv=config.NT_TRANS_MAT_CSV):
        super().__init__()
        self.L = L
        self.model_name = model_name
        self.WT = config.WT[:self.L]
        self.WT_ONE_HOT = torch.eye(self.qc, dtype=self.PRECISION)[self.WT]

        self.counts_main_d, self.counts_int_d = \
                calc_count_data(msa_filenames_dict, max_seq_length=L)

        logging.info(f"Size of round 15 interaction tensor : "
                     f"{ utils.calc_tensor_size(self.counts_int_d[15])}")
        self.num_seqsd = calc_num_seqsd(self.counts_main_d)

        logging.info(f"Reading trans_mat csv : {nt_trans_mat_csv}")
        self.trans_mat, self.double_trans_mat = \
                calc_transition_matrices(nt_trans_mat_csv)

        logging.info(f"trans_mat.shape : {self.trans_mat.shape}")
        logging.info(f"double_trans_mat.shape = {self.double_trans_mat.shape}")

    def forward(self):
        raise NotImplementedError

    def calc_total_reg(self):
        """ Derived classes should supply this function """
        raise NotImplementedError

class PairwiseModel(EvolutionCodonModel):

    def __init__(self, 
                 lam_main=0.0, # regularization for main effects
                 lam_int=0.0, # regularization for interaction effects
                 margin_penalty=0.0, # penalty for marginalization constraints
                 conditional=False, # conditional likelihood on prev rounds
                 include_main_params=False, # Marginalize interaction effects 
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.lam_int = lam_int
        self.lam_main = lam_main
        self.margin_penalty = margin_penalty
        self.conditional = conditional

        main_params, int_params = self.get_initial_params()
        logging.info(f"Interaction params Shape : {int_params.shape}")

        # with the initialized weights we should not have any margin loss
        logging.info(f"Margin Loss with initialization values : "
                     f"{calc_margin_loss(main_params, int_params)}")
        logging.info(f"Checking symmetry of interaction parameters : "
                 f"{(int_params - int_params.permute(2,3,0,1)).abs().max()}")

        self.include_main_params = include_main_params
        if self.include_main_params: 
            logging.info(f"Including main effect parameters")
            self.main_params = torch.nn.Parameter(main_params)
        else:
            logging.info(f"Removing main effect parameters")
            self.main_params = None
        self.int_params = torch.nn.Parameter(int_params)


    def get_initial_params(self):
        """ Convert the last round counts to frequencies and then take log
            This will be the starting parameters
        """
        # start parameters close to the energy values of the last round
        last_round = max(self.counts_int_d.keys()) # integer for round number
        last_round_N = self.num_seqsd[last_round]

        # initialization
        # Here we add a pseudo count so that we do not encounter log of zero
        # anywhere.  since these parameters are going to get optimized it
        # doesn't matter what pseudo_count we pick
        pseudo_count = 1
        initial_freqs_main = (self.counts_main_d[last_round]  # shape (L, qc)
                + pseudo_count / self.qc).to(dtype=self.PRECISION) \
                             / (last_round_N + pseudo_count)
        initial_freqs_int = (self.counts_int_d[last_round] # shape (L,qc,L,qc)
                + pseudo_count / (self.qc*self.qc)).to(dtype=self.PRECISION) \
                             / (last_round_N + pseudo_count)

        # project frequencies down to Amino Acid level
        # then convert to energy via log (over here larger is fitter)
        main_params = (torch.einsum("ic,ca->ia", initial_freqs_main,
                        self.CODON_AA_MATRIX)
                            + self.EPSILON).log() # shape (L, qa)
        int_params = (torch.einsum("icjd,ca,db->iajb", initial_freqs_int,
                        self.CODON_AA_MATRIX, self.CODON_AA_MATRIX)
                            + self.EPSILON).log()  # shape (L, qa, L, qa)

        del initial_freqs_main, initial_freqs_int

        # symmetrize interaction parameters
        int_params = symmetrize_int_params(int_params)
        int_params = zero_diagonal_int_params(int_params)

        return main_params, int_params


    def calc_wt_int_ind(self):
        # WT double indicator vector
        # This is the 2D version of WT_ONE_HOT
        return torch.einsum("ia,jb->iajb",
                                self.WT_ONE_HOT, self.WT_ONE_HOT)

    def forward(self):
        ll = torch.zeros(1)
        N = sum(self.num_seqsd.values())
        ## project parameters

        # 1. interaction parameters should be symmetric so let us project them
        # Also the diagonal elements int_params(i, *, i, *) should be zero
        # However, we ignore them now and zero it out only when computing the
        # log likelihood.
        #
        # FIXME: Should we zero-diagonal here also?  First check if the
        # gradient is zero.
        # 2. Exponentiate to make positive so that parameters represent fitness
        # 3. Normalize into a probability vector
        if self.include_main_params:
            main_params_prob = convert_main_fitness_to_probs(
                                                    self.main_params.exp())
            # project params to codon level
            main_params_prob = main_params_prob.index_select(1, 
                                                    self.CODON_AA_MAP)
        int_params_prob = convert_int_fitness_to_probs(
                                symmetrize_int_params(self.int_params).exp())
        # project params to codon level
        int_params_prob = int_params_prob.index_select(1,
                       self.CODON_AA_MAP).index_select(3, self.CODON_AA_MAP)

        # Now calculated log likelihood for each round that we have data for
        for round_num in range(1, max(self.counts_int_d.keys())+1):
            if round_num == 1: # Initialize with round 0 values
                if self.include_main_params:
                    main_sorted_probs = self.WT_ONE_HOT
                int_sorted_probs = self.calc_wt_int_ind()
            # index t: codon transition `to`
            # index f: codon transition `from`
            # index i: residue position number
            if self.include_main_params:
                main_unsorted_probs = torch.einsum("tf,if->it", self.trans_mat,
                                          main_sorted_probs)
                main_sorted_probs = convert_main_fitness_to_probs(
                                        main_unsorted_probs * main_params_prob)
            # index o: second codon transition `to`
            # index r: second codon transition `from`
            # index j: second residue position number
            int_unsorted_probs = torch.einsum("tfor,ifjr->itjo",
                                    self.double_trans_mat, int_sorted_probs)
            int_sorted_probs = convert_int_fitness_to_probs(
                                    int_unsorted_probs * int_params_prob)

            if round_num in self.counts_int_d:
                if not self.include_main_params:
                    main_sorted_probs = marginalize_int_probs(int_unsorted_probs)
                ll_round_main = ((main_sorted_probs).log() *
                            self.counts_main_d[round_num]).sum()
                ll_round_int_full = ((int_sorted_probs).log() *
                            self.counts_int_d[round_num]).sum(axis=(1,3))

                # Remove main effects from interaction effects (by zeroing out
                # the diagonal) then we can sum to get the log likelihood
                ll_round_int = (ll_round_int_full *
                                        (1 - torch.eye(self.L))).sum()
                ll_round = ll_round_main + ll_round_int
                if torch.any(torch.isnan(ll_round)):
                    logging.error("Found NaNs in log likelihood")
                ll += ll_round # / num_seqsd[round_num]
                # FIXME: should we add an option to normalize
                # between the rounds for sequencing depth?
                if self.conditional: # reset the probabilites to counts
                    raise NotImplementedError
        return ll

    def calc_margin_loss(self):
        """ Penalty to enforce marginalization constraints """
        return self.margin_penalty * calc_margin_loss(self.main_params, 
                                                        self.int_params)

    def calc_main_reg(self):
        """ Main effects regularization """
        ret = torch.tensor(0.)
        if self.include_main_params:
            ret = self.lam_main * calc_l2(self.main_params)
        return ret

    def calc_int_reg(self):
        """ Interaction effects regularization """
        return self.lam_int * calc_l2(zero_diagonal_int_params(self.int_params))

    def calc_total_reg(self):
        """ Sum of all the regularizers and penalties """
        return self.calc_margin_loss() + self.calc_main_reg() + \
                        self.calc_int_reg()

    @property
    def margin_norm(self):
        return calc_margin_loss(self.main_params, self.int_params).sqrt()

    @property
    def main_norm(self):
        ret = None
        if self.include_main_params:
            ret = self.main_params.norm().item()
        else:
            ret = self.int_params.exp().sum(axis=(1,3)).log().norm().item()
        return ret

    @property
    def int_norm(self):
        return self.int_params.norm().item()

    @property
    def int_min(self):
        return self.int_params.min().item()

    def save_params(self, savedir=config.WORKING_DIR):
        L = self.L
        prefix = self.model_name
        torch.save(self.int_params, f"{savedir}/{prefix}_{L}_int_params.pt")
        if self.include_main_params:
            torch.save(self.main_params, 
                                    f"{savedir}/{prefix}_{L}_main_params.pt")


def run_optimizer(ecm, num_epochs=300, lr=0.01, verbose=True):
    """
        Optimize an EvolutionCodonModel
    """
    optimizer = torch.optim.Adam(ecm.parameters(), lr = lr)
    losses = np.zeros(num_epochs, dtype=np.double)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        nll = - ecm()   # This calls the forward function
        loss = nll  + ecm.calc_total_reg()
        loss.backward()
        losses[epoch] = loss.item()
        if verbose and not (epoch % 10):
            with torch.no_grad():
                print(f"Epoch: {epoch:03d}, "
                      f" Pnll: {losses[epoch]:.2f}, "
                      f"Margin Norm: {ecm.margin_norm:.2f}, "
                      f"Main Norm: {ecm.main_norm:.2f}, "
                      f"Int Norm: {ecm.int_norm:.2f}, "
                      f"Int Min: {ecm.int_min:.2f}", flush=True)
        optimizer.step()
    return losses


def save_losses(losses, prefix="", savedir=config.WORKING_DIR):
    """ `losses`: numpy array of losses """
    np.save(f"{savedir}/{prefix}losses.npy", losses)
    plt.plot(losses)
    plt.title("Loss Curve for minimizing negative log likelihood with penalty")
    plt.xlabel("Optimizer Step number")
    plt.ylabel("Loss")
    plt.savefig(f"{savedir}/{prefix}loss.png");


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate",
            help="Learning rate for ADAM optimizer",
            type=float, default=0.03)
    parser.add_argument("-j", "--disable_joblib",
            help="Disable saving of joblib cache",
            action='store_true')
    parser.add_argument("-n", "--num_epochs",
            help="Number of training steps",
            type=int, default=300)
    parser.add_argument("-m", "--model_name",
            help="Model Name",
            type=str, default="DHFR")
    parser.add_argument("-v", "--log_level",
            help="Log level for debugging messages",
            type=str, default="INFO")
    parser.add_argument("-o", "--output_directory",
            help="Output directory for model params and figs"
                 " (Path Relative to the scripts directory)",
            default=config.WORKING_DIR)
    parser.add_argument("-t", "--nt_trans_mat_csv",
            help="Nucleotide Transition Matrix pickle file "
                 " (Path Relative to the scripts directory)",
            default=config.NT_TRANS_MAT_CSV)
    parser.add_argument("-L", "--protein_length",
            help="Change protein length for debugging",
            type=int, default=config.L)
    parser.add_argument("--lam_main",
            help="Regularization for main effects parameters",
            type=float, default=1e-3)
    parser.add_argument("--lam_int",
            help="Regularization for interaction effects parameters",
            type=float, default=1e-4)
    parser.add_argument("-i", "--include_main_params",
            help="Include Main Parameters",
            action='store_true')
    parser.add_argument("--margin_penalty",
            help="Penalty for margin loss",
            type=float, default=100000.)
    args = parser.parse_args()

    # set the log level
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.disable_joblib:
        config.disable_joblib_memory()
        utils.undo_joblib_memory()

    logging.info("calc_mutant_counts signature : %s",
                    utils.calc_mutant_counts)

    start_time = time.time()
    ecm = PairwiseModel(
                      model_name = args.model_name,
                      L = args.protein_length,
                      lam_main = args.lam_main,
                      lam_int = args.lam_int,
                      margin_penalty = args.margin_penalty,
                      nt_trans_mat_csv = args.nt_trans_mat_csv,
                      include_main_params = args.include_main_params)
    logging.info("Finished Initializing PairwiseModel")
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    start_time = time.time() 
    ### Optimization
    losses = run_optimizer(ecm, num_epochs=args.num_epochs, 
                            lr=args.learning_rate)
    save_losses(losses, prefix=f"{ecm.model_name}_",
                            savedir=args.output_directory)
    ecm.save_params(savedir=args.output_directory)
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    with torch.no_grad():
        print(f"Main_params Norm: {ecm.main_norm:.4f}")
        print(f"Int params Norm: {ecm.int_norm:.4f}")
