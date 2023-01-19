"""
Created on Tue Apr 16 10:01:36 2019

@author: emily
"""
import pandas as pd

# Phred33 Qscore mapping
Qscore = dict((chr(i),i-33) for i in range(33,127))
Ascii = dict((i, chr(i+33)) for i in range(0,94))

# function parse_SAM written by Phil Romero (NGS_tools.py),
# modified by Emily
def parse_SAM(read,reference):
    """ read -> a single line from a SAM file
        reference -> the reference sequence
        
        Returns the name of the sequence and the alignment to the
        reference based on the input read. """
        
    # Organize the read data
    data = read.split('\t')
    name = data[0].strip()
    start = int(data[3])-1 # convert to 0-based index
    flag = int(data[1])
    cigar = data[5]
    seq = data[9]
    Q = [Qscore[q] for q in data[10]]
    
    if flag==4: return '',''

    # Split the CIGAR string into sections (values in clist)
    clist, pos = [], ''
    for p in cigar:
        if p.isalpha():
            clist.append((int(pos),p))
            pos = ''
        else: pos+= p
        
    # Clip the sequence and corresponding Q values according to clist
    if clist[0][1]=='S':
        seq = seq[clist[0][0]:]
        Q = Q[clist[0][0]:]
        clist.pop(0)
    if clist[-1][1]=='S':
        seq = seq[:-clist[-1][0]]
        Q = Q[:-clist[-1][0]]
        clist.pop(-1)
        
    # Cycle through the rest of clist and generate the alignment
    refstart, readstart = start, 0
    alignment = [(reference[i],'N',40) for i in range(start)] # First part of alignment = reference and Ns
    for section in clist:
        if section[1]=='M': # Match: add to both
            ref = reference[refstart:refstart+section[0]]
            read = seq[readstart:readstart+section[0]]
            readQ = Q[readstart:readstart+section[0]]

            alignment.extend(tuple(zip(ref,read,readQ)))
            
            refstart += section[0]
            readstart += section[0]
            
        elif section[1]=='D': # deletion: only add to ref seq
            ref = reference[refstart:refstart+section[0]]
            read = '-'*section[0]
            readQ = '-'*section[0]
            
            alignment.extend(tuple(zip(ref,read,readQ)))
            
            refstart += section[0]
        
        elif section[1]=='I': # insertion: only add to read seq
            ref = '-'*section[0]
            read = seq[readstart:readstart+section[0]]
            readQ = Q[readstart:readstart+section[0]]
            
            alignment.extend(tuple(zip(ref,read,readQ)))
            
            readstart += section[0]
            
    alignment.extend([(reference[i],'N',40) for i in range(refstart,len(reference))]) # add on rest of reference sequence
        
    # Need to assign Qscores for deletions: assume good enough + filter out later
    deletions = [i for i in range(len(alignment)) if alignment[i][2]=='-']
    for pos in deletions:
        delQ = 40
        alignment[pos] = alignment[pos][:2]+(delQ,)
    
    return name, alignment

def parse_fasta(fastafile):
    """ Obtain the list of amino-acid sequences (without associated identifiers)
        present in a FASTA file.
        
        fastafile -> name of FASTA file from which you want to extract sequences
        
        Returns list of sequences contained in FASTA file
    """
    
    seqs = []
    curr_seq = ''
    with open(fastafile, 'r') as inf:
        for line in inf:
            line = line.strip()
            if line[0]=='>':
                if len(curr_seq)>0:
                    seqs.append(curr_seq)
                    curr_seq = ''
            elif len(line)>0: curr_seq += line         
        if len(curr_seq)>0: seqs.append(curr_seq)
    
    return seqs

def aln_to_fastq(name, alignment, start, end):
    """ Formats a parsed SAM file into a FASTQ file suitable for downstream
        NGS analyses.
        
        name-> identifier associated with read
        alignment -> aligned reference and read sequence with associated q scores
                as output by the parse_sam function
        start -> starting position of alignment/reference sequence indexing that
                corresponds to position desired to be written in FASTQ
        end -> last position of alignment/reference sequence indexing that
                corresponds to the last position desired to be written in FASTQ
                
        Returns string with formatted FASTQ entry for the input read sequence
    """
    
    seq = ''
    Qvals = ''
    all_gaps = True
    for i in range(len(alignment[start:end])):
        ref, read, q = alignment[start:end][i]
        # if any insertions, deletions, ambiguity, then exclude sequence
        if ref == '-' or read == '-' : return ''
        
        if read == 'N': 
            read = '-' # For FASTQ purposes, introdruce gap to distinguish from N
            q == 40
        else: all_gaps = False # as long as we find some non-N non insert, non-deleted residue, we have a non-gapped segment

        seq += read
        Qvals += Ascii[q]
    
    if all_gaps: return '' # ignore sequences where whole gene segment consists of gaps

    fastq = ["@%s"%name,
             seq,
             '+',
             Qvals]
    
    return '\n'.join(fastq)

def sam_to_aligned_fastq(samfile, reference, outfastq, start=0, end=-1):
    """ Takes a given SAM file and the corresponding reference sequence for the
        alignment and converts into a FASTQ file for downstream NGS analysis.
        
        samfile -> file name of the SAM read alignment
        reference -> the full sequence used as the reference for producing the SAM alignment
        outfastq -> file to which the output FASTQ should be written
        start -> first position index from the reference sequence/alignment desired
            to be written to FASTQ
        end -> last position index from the reference sequence/alignment desired to
            be written to FASTQ
            
        Returns None
    """

    refseq = parse_fasta(reference)
    
    if len(refseq)!=1:
        print("sam_to_aligned_fastq: Failed to convert SAM to FASTQ: more than one reference sequence")
        return
    
    refseq = refseq[0]
    if end<0: end=len(refseq)
    
    fastqs = []
    with open(samfile, 'r') as inf:
        for line in inf:
            line = line.strip()
            if line[0]=='@': continue
        
            name, alignment = parse_SAM(line, refseq)
            if len(alignment)==0: continue
            fastq_line = aln_to_fastq(name, alignment, start, end)
            if len(fastq_line)>0: fastqs.append(fastq_line)
            
    with open(outfastq, 'w') as outf:
        outf.write('\n'.join(fastqs))

    return

def remove_seqs(exclude, oldfastq, newfastq='new.fastq'):
    ''' exclude -> array-like of sequence IDs
        oldfastq -> file containing list of sequences in fastq format (unaligned) 
        newfastq -> file to write new list of sequences in fastq format (unaligned)
        
        Writes FASTQ entries from file oldfastq to newfastq if and only if
        they are not in exclude. '''
      
    # Make the list of excluded sequences hashable
    try:
        if len(exclude)>0:
            exclude = set(exclude)
    except TypeError:
        exclude = set()
    
    print("%d in SET of sequences to exclude" % len(exclude))

    # Get the desired FASTQ sequences
    write_lines = []
    excluded = 0
    with open(oldfastq, 'r') as inf:     
        k = 0
        write = True
        for line in inf:
            # Parse FASTQ
            if k==0 and line[0]=='@': # sequence identifier
                if line[1:].strip() in exclude: 
                    write = False
                    excluded += 1
                if write: write_lines.append(line)
                k += 1
            elif k==1: # nucleotide sequence
                if write: write_lines.append(line)
                k += 1
            elif k==2: # spacer line
                if write: write_lines.append(line)
                k += 1
            elif k==3: # quality encoding
                if write: write_lines.append(line)
                write = True
                k = 0
     
    print("%d sequences successfully excluded." % excluded)

    # Total number of lines should be divisible by 4 (four lines per sequence)
    if len(write_lines)%4!=0:
        print("remove_seqs: Failed to properly parse FASTQ. No new file written.")
        return
    
    # Write the output
    with open(newfastq, 'w') as outf:
        outf.write('\n'.join(write_lines))
        
    return

def filter_ccs(ccs_statistics, min_passes = 10, min_length = 564, min_quality = 0.99, exclude=True):
    ''' ccs_statistics -> file in CSV format containing CCS name, number of passes,
                            CCS length, and read score for each CCS in a PacBio run
        min_passes -> minimum number of passes in order for a CCS to be kept.
                        Default is 10, which should correspond to ~99% accuracy
        min_length -> minimum length of read in order for CCS to be kept. Default
                        is 564, which is the length of my coding mDHFR gene.
        min_quality -> minmium read score in order for CCS to be kept. Default is
                        0.99, which is (relatively) arbitrary but seems like it should
                        be pretty good...
        exclude -> If True, this function yields a list of CCS to REMOVE. If False,
                    the returned list specifies the sequences to keep. Default True.
                        
        Returns a list of CCS identifiers that fail (if exclude=True) or pass (if exclude=False)
        the specified filtering.
    '''
    
    # ensure file is CSV
    if ccs_statistics.strip().split('.')[-1] != 'csv':
        print("CCS Statistics file format not recognized. Please make sure \
              CSV file formatting is used.")
    
    # read the CSV into a dataframe
    df = pd.read_csv(ccs_statistics)
    
    if not exclude:
        df = df[(df.npasses>=min_passes) & (df.qlen>=min_length) & (df.readscore>=min_quality)]
    else:
        df = df[(df.npasses<min_passes) | (df.qlen<min_length) | (df.readscore<min_quality)]
    
    return list(df.qname)

#---- at some other point you can define a main function ----#

# Uncomment for filtering
'''
exclude = filter_ccs("ccs_statistics.csv")
print("%d sequences should be excluded from full FASTQ" % len(exclude))
remove_seqs(exclude, "ccs.fastq", newfastq='filtered_ccs.fastq')
'''

# Uncomment for SAM parsing
sam_to_aligned_fastq("./190410_all_reads_aligned_ccs.sam", "./SacI_NdeI_pET22hc-mDHFR.fa", "all_reads_aligned_fastq_190510.fastq", 130, 694)
