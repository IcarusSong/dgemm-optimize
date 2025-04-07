# Compiler and flags
CC = gcc
CFLAGS = -O3 -mavx512f  -fopenmp -Wextra -I./include
LDFLAGS = -lopenblas -lm

# Source files
SRC_DIR = .
DGEMM_SRC = dgemm/dgemm.c
KERNEL_SRC = kernel/kernel.c
MAIN_SRC = main.c

# Object files
OBJS = $(DGEMM_SRC:.c=.o) $(KERNEL_SRC:.c=.o) $(MAIN_SRC:.c=.o)

# Executable name
TARGET = dgemm_run

# Default target
all: $(TARGET) run

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

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
