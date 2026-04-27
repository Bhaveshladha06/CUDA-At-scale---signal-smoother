NVCC     ?= nvcc

SRC_DIR  := src
BIN_DIR  := bin
TARGET   := $(BIN_DIR)/batch_signal_smoother
SRC      := $(SRC_DIR)/batchSignalSmoother.cu

# -O3 optimisation; math is all float so --use_fast_math is safe here
NVCCFLAGS := -O3 -std=c++14 --use_fast_math -lm

.PHONY: all clean help run

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./run.sh

clean:
	rm -f  $(TARGET)
	rm -rf data/output/*
	rm -rf data/input/*

help:
	@echo "make        Build GPU batch signal smoother"
	@echo "make run    Build then execute default workload (200 signals, 1024 samples)"
	@echo "make clean  Remove built binary and all generated data"
