ROOT_DIR:=./
SRC_DIR:=./src
test:
	PYTHONPATH=$(SRC_DIR) pytest

.PHONY: test
