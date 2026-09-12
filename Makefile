SEEDS = 0 1 2 3 4 5 6 7 8 9
PY = python

.PHONY: all tests phase1 merge clean
all: tests phase1 merge

tests:
	$(PY) tests.py

phase1:
	@for s in $(SEEDS); do $(PY) run.py --phase 1 --seed $$s; done

merge:
	$(PY) run.py --merge

clean:
	rm -f results/phase*_seed*.parquet results.parquet
