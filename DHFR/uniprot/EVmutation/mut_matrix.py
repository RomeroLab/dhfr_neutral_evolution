import sys

from model import CouplingsModel
import tools

c = CouplingsModel("mDHFR.model_params")

singles = tools.single_mutant_matrix(
    c, output_column="effect_prediction_epistatic"
)

singles.to_csv("mDHFR_EVmutation.csv", index=False)



