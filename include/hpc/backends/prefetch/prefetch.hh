#pragma once

// __builtin_prefetch API

#define PREFETCH_READ_L1 (addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_READ_L2 (addr) __builtin_prefetch((addr), 0, 2)
#define PREFETCH_READ_L3 (addr) __builtin_prefetch((addr), 0, 1)
#define PREFETCH_READ_NTA (addr) __builtin_prefetch((addr), 0, 0)
#define PREFETCH_WRITE_L1 (addr) __builtin_prefetch((addr), 1, 3)
#define PREFETCH_WRITE_L2 (addr) __builtin_prefetch((addr), 1, 2)
#define PREFETCH_WRITE_L3 (addr) __builtin_prefetch((addr), 1, 1)
#define PREFETCH_WRITE_NTA (addr) __builtin_prefetch((addr), 1, 0)
