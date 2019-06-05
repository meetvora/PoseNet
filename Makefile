CPU   := 6
Q     := 24
SCRIPT:= main.py

autoformat:
	yapf -ir --style pep8 .

clean:
	rm lsf*
	rm *.log

requirements:
	pip install -r requirements.txt

submit:
	bsub -n $(CPU) -W $(Q) -R "rusage[mem=18000, ngpus_excl_p=1]" python $(SCRIPT)

status:
	watch -n 5 bjobs

%:
	@: