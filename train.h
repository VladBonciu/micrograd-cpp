#pragma once

#include <chrono>
#include <stdio.h>

#include "Value.h"
#include "Network.h"
#include "BatchReadCSV.h"

#define N_INPUT 4
#define N_OUTPUT 3

#define N_EPOCHS 16
#define BATCH_SIZE 64

#define LEARNING_RATE 0.01

#define MOMENTUM_COEFFICIENT 0.9
