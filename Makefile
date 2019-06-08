CPU   := 4
Q     := 24:00
SCRIPT:= main.py
DIR   := master

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

stream:
	tail -200f log/$(DIR)/`(ls -t log/$(DIR) | head -1)`
%:
	@:
