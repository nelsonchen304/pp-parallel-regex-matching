.PHONY: all clean report benchmark

CC = g++
NVCC = nvcc
COMMON_FLAGS = -lpcap -Iinclude
CC_FLAGS = -fPIC -fopenmp -Wall -Wextra -Werror
CUDA_COMPILE_FLAGS = -arch=sm_86 -Xcompiler '$(CC_FLAGS)'
CUDA_LINK_FLAGS = -rdc=true -arch=sm_86 -Xcompiler '$(CC_FLAGS)'

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -Og -g
else
	COMMON_FLAGS += -O3
endif

SRC_DIR = src
LIB_DIR = lib
BUILD_DIR = build
BIN_DIR = bin

SRCS = $(wildcard $(LIB_DIR)/*.cpp)
CUDA_SRCS = $(wildcard $(LIB_DIR)/*.cu)

OBJS = $(patsubst $(LIB_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
OBJS += $(patsubst $(LIB_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRCS))
DFA_OBJS = $(BUILD_DIR)/regex.o

OUT_PP = pp-parallel-regex
OUT_DFA = regex2dfa

all: $(BIN_DIR)/$(OUT_PP) $(BIN_DIR)/$(OUT_DFA)

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/$(OUT_PP): $(BUILD_DIR)/$(OUT_PP).o $(OBJS) | $(BIN_DIR)
	$(NVCC) -o $@ $^ $(CUDA_LINK_FLAGS) $(COMMON_FLAGS)

$(BIN_DIR)/$(OUT_DFA): $(BUILD_DIR)/$(OUT_DFA).o $(DFA_OBJS) | $(BIN_DIR)
	$(CC) -o $@ $^ $(CC_FLAGS) $(COMMON_FLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CC) -c $< -o $@ $(CC_FLAGS) $(COMMON_FLAGS)

$(BUILD_DIR)/%.o: $(LIB_DIR)/%.cpp | $(BUILD_DIR)
	$(CC) -c $< -o $@ $(CC_FLAGS) $(COMMON_FLAGS)

$(BUILD_DIR)/%.o: $(LIB_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(CUDA_COMPILE_FLAGS) $(COMMON_FLAGS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

benchmark: $(BIN_DIR)/$(OUT_PP)
	uv run scripts/benchmark.py

report:
	make -C report/
