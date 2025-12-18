# Makefile for UCVME Docker operations

.PHONY: build run train test shell clean help

# Default values
DATASET ?= so2sat_pop
DATA_ROOT ?= /work/ammar/sslrp/data
OUTPUT ?= ./outputs/experiment
RD_LABEL ?= 1000
RD_UNLABEL ?= 5000

help:
	@echo "UCVME Docker Commands:"
	@echo "  make build          - Build Docker image"
	@echo "  make run            - Run interactive shell"
	@echo "  make train          - Run training (use DATASET, OUTPUT, etc. to customize)"
	@echo "  make test           - Run testing only"
	@echo "  make shell          - Open interactive shell"
	@echo "  make clean          - Clean up Docker containers and images"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATASET=so2sat_pop OUTPUT=./outputs/exp1"
	@echo "  make train RD_LABEL=500 RD_UNLABEL=2000"

build:
	docker-compose build

run:
	docker-compose run --rm ucvme /bin/bash

train:
	@echo "Training with dataset: $(DATASET)"
	@echo "Output: $(OUTPUT)"
	mkdir -p $(OUTPUT)
	docker-compose run --rm ucvme python3 ucvme.py \
		--dataset $(DATASET) \
		--data_root /data \
		--output /workspace/outputs/$(shell basename $(OUTPUT)) \
		--rd_label $(RD_LABEL) \
		--rd_unlabel $(RD_UNLABEL) \
		--num_epochs 30 \
		--batch_size 32

test:
	@echo "Testing with dataset: $(DATASET)"
	docker-compose run --rm ucvme python3 ucvme.py \
		--dataset $(DATASET) \
		--data_root /data \
		--output /workspace/outputs/$(shell basename $(OUTPUT)) \
		--test_only \
		--weights /workspace/outputs/$(shell basename $(OUTPUT))/best.pt

shell:
	docker-compose run --rm ucvme /bin/bash

clean:
	docker-compose down
	docker system prune -f

clean-all: clean
	docker rmi ucvme:latest || true

