#!/bin/bash

MSATOOLS=~/VAEs/bin/msa_tools.sh
echo $MSATOOLS -i ../Gen15_aa.aln.gz -r Gen15_remove.aln.gz -w ../../../mDHFR_WT.aln 
echo $MSATOOLS -i ../Gen15_aa.aln.gz -r Gen15_wt_remove.aln.gz -w ../../../mDHFR_WT.aln -a


CONV_ALN_PY=convert_alignment_archive_to_a2m.py
echo python $CONV_ALN_PY Gen15_remove.aln.gz Gen15_remove.a2m
echo python $CONV_ALN_PY Gen15_wt_remove.aln.gz Gen15_wt_remove.a2m


PLMC_BIN=~/software/plmc/bin/plmc
theta=0.09

PLMC_ADD_ARGS="-m 150 -lh 1.0 -le 37.0 -lg 0.0 -n 20"

echo $PLMC_BIN \
    -c plmc_Gen15_remove.txt \
    --theta ${theta} \
    ${PLMC_ADD_ARGS} \
    Gen15_remove.a2m
rm -f Gen15_remove.a2m

echo $PLMC_BIN \
    -c plmc_Gen15_wt_remove.txt \
    ${PLMC_ADD_ARGS} \
    --theta ${theta} \
    Gen15_wt_remove.a2m
rm -f Gen15_wt_remove.a2m

python $CONV_ALN_PY ../Gen15_aa.aln.gz  Gen15_untouched.a2m
$PLMC_BIN \
        -c plmc_Gen15_untouched.txt \
        ${PLMC_ADD_ARGS} \
        --theta 0.0001 \
        Gen15_untouched.a2m
rm -f Gen15_untouched.a2m

