# Compiler and flags
CC = gcc
CFLAGS = -O3 -mavx512f -fopenmp -Wextra -I./include -MMD -MP  # 添加 -MMD 和 -MP
LDFLAGS = -lopenblas -lm

# Source files
SRC_DIR = .
DGEMM_SRC = dgemm/dgemm.c
KERNEL_SRC = kernel/kernel.c
MAIN_SRC = main.c

# Object files
OBJS = $(DGEMM_SRC:.c=.o) $(KERNEL_SRC:.c=.o) $(MAIN_SRC:.c=.o)

# Dependency files (.d files)
DEPS = $(OBJS:.o=.d)

# Directories
BIN_DIR = bin

# Executable name
TARGET = $(BIN_DIR)/dgemm_run

# Default target
all: $(TARGET) run

# Ensure bin directory exists
$(shell mkdir -p $(BIN_DIR))

# Link all object files to create executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Run the program
run: $(TARGET)
	@echo "Running program..."
	@./$(TARGET)

# Compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Include generated dependency files
-include $(DEPS)

# Clean up
clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

.PHONY: all clean run