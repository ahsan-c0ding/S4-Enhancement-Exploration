# Makefile for Milestone 3 RISC-V Assembly (S4D Galaxy Classifier)
# This Makefile is intended to be run from the root of the riscv-env-setup directory.

CC = riscv32-unknown-elf-gcc
CFLAGS = -march=rv32gcv -mabi=ilp32f -nostartfiles -lm
LDFLAGS = -T veer/link.ld

BUILD_DIR = build/exe

SRCS = main.s math.s nn.s
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/math.o $(BUILD_DIR)/nn.o

TARGET = $(BUILD_DIR)/galaxy_classifier.exe

all: $(TARGET)

$(BUILD_DIR)/%.o: %.s
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET)
	@echo "Build successful! Executable is at $(TARGET)"

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."

.PHONY: all clean