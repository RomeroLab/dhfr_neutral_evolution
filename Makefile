
nothing:
	@echo

CONDA_BLANK=

.PHONY: check_no_conda 
check_no_conda:
ifneq ($(CONDA_DEFAULT_ENV), $(CONDA_BLANK))
	$(error CONDA needs to be deactivated)
endif

.PHONY: install_conda 
install_conda: check_no_conda
	@conda env create -f environment.yml

.PHONY: delete_conda
delete_conda: check_no_conda
	@conda remove --name dhfr_analysis --all -y

.PHONY: reinstall_conda
reinstall_conda: delete_conda install_conda

.PHONY: ext
ext:
	$(MAKE) -C scripts 

.PHONY: clean_logs
clean_logs:
	rm -f working/logs/chtc_run_*.txt

.PHONY: clean
clean:
	rm -f docker_stderror
	rm -f staging.tar.gz	

.PHONY: deepclean
deepclean: clean clean_logs
	rm -f output_*.tar.gz

.PHONY: verydeepclean
verydeepclean: deepclean
	$(MAKE) -C scripts clean

.PHONY: staging
staging: staging.tar.gz

staging.tar.gz : 
	tar --exclude='.git' \
			--exclude='.gitignore' \
			--exclude='htc.sh' \
			-czvf staging.tar.gz \
			./scripts \
			./DHFR \
			./working/README.md \
			Makefile \
			*.sh 

submit: staging
	condor_submit DISABLE_GPU=1 DISABLE_FLOCKING=1 htc.sub

flocking: staging
	condor_submit DISABLE_GPU=1 htc.sub

environment.explicit.txt:
	conda list --explicit > environment.explicit.txt

