# ABOUTME: One command to rebuild everything: lint, tests, then every notebook in order.
# ABOUTME: Notebooks run in place so results/ and figures/ are regenerated from a clean kernel.
NOTEBOOKS := $(sort $(wildcard notebooks/[0-9]*.ipynb))

.PHONY: all lint test notebooks

all: lint test notebooks

lint:
	ruff check .

test:
	pytest -q

notebooks:
	@for nb in $(NOTEBOOKS); do \
		echo "Running $$nb"; \
		jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
