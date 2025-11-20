# Agent Build Guide for amp Repository

This guide explains how to build and test the C/C++ native code in the amp repository.

## Repository Structure

The main C/C++ codebase is located in:
- **`src/native/`** - Native C/C++ implementation
  - `amp_kernels.c` - Core kernel implementations (compiled as C++)
  - `fft_backend.cpp` - FFT backend interface
  - `graph_runtime.cpp` - Graph execution runtime
  - `include/` - Public header files
  - `nodes/` - Node-specific implementations
    - `fft_division/` - FFT division node (included via .inc files)
  - `tests/` - Native unit tests

## Build System

The project uses **CMake** as its build system. The main CMakeLists.txt is at:
```
src/native/CMakeLists.txt
```

## Dependencies

The build requires:
1. **CMake** >= 3.18
2. **C++17** compatible compiler
3. **Eigen3** (automatically fetched if not present)
4. **fftfree** library (fetched from https://github.com/sangderenard/fftfree.git)

## Building the Code

### Step 1: Create Build Directory
```bash
cd /path/to/amp
mkdir -p build
cd build
```

### Step 2: Configure with CMake
```bash
# Debug build (recommended for development)
cmake ../src/native -DCMAKE_BUILD_TYPE=Debug

# Release build
cmake ../src/native -DCMAKE_BUILD_TYPE=Release
```

### Step 3: Build
```bash
# Build all targets
cmake --build . --config Debug

# Or use make directly on Unix-like systems
make -j$(nproc)

# On Windows with Visual Studio
cmake --build . --config Debug
```

### Build Outputs
Compiled binaries are placed in:
- **Single-config generators (Unix Makefiles):** `build/`
- **Multi-config generators (Visual Studio):** `build/Debug/` or `build/Release/`

## Running Tests

### Run All Tests
```bash
cd build
ctest -C Debug --output-on-failure
```

### Run Specific Test
```bash
# From the build directory
./Debug/test_fft_division_node    # Windows
./test_fft_division_node          # Unix-like

# With options
./test_fft_division_node --help
./test_fft_division_node --quiet
./test_fft_division_node --verbosity trace
./test_fft_division_node --hop 2 --overlap 0.5
```

## Key Source Files for FFT Division Node

The FFT division node issue is related to these files:

1. **State Structure Definition:**
   - `src/native/amp_kernels.c` (lines ~1820-1968)
   - Contains `struct { ... } fftdiv;` with node state

2. **Node Implementation:**
   - `src/native/nodes/fft_division/fft_division_nodes.inc`
   - Included by amp_kernels.c
   - Contains `run_fft_division_node()` and execution logic

3. **State Management:**
   - `ensure_fft_stream_slots()` in amp_kernels.c (~line 3463)
   - Creates forward/inverse FFT handles

4. **Lane Planning:**
   - `fftdiv_prepare_lane_plan()` in fft_division_nodes.inc (~line 1220)
   - Determines which stages are active

5. **Test:**
   - `src/native/tests/test_fft_division_node.cpp`
   - Comprehensive test including spectral mode

## Build Options

### Enable Logging
```bash
cmake ../src/native -DAMP_NATIVE_ENABLE_LOGGING=ON
```

### Disable Tests
```bash
cmake ../src/native -DAMP_NATIVE_BUILD_TESTS=OFF
```

### Custom Compile/Link Flags
```bash
export AMP_NATIVE_EXTRA_COMPILE_ARGS="-Wall -Wextra"
export AMP_NATIVE_EXTRA_LINK_ARGS="-fsanitize=address"
cmake ../src/native
```

## Troubleshooting

### Network Issues (Eigen/fftfree fetch failures)
If you're in a restricted network environment:
1. Clone dependencies manually:
   ```bash
   mkdir -p third_party
   cd third_party
   git clone https://github.com/sangderenard/fftfree.git fftfree
   git clone https://gitlab.com/libeigen/eigen.git eigen
   cd ..
   ```
2. CMake will detect and use the local copies

### Compilation Errors
- Ensure C++17 support: `g++ --version` or `clang++ --version`
- On older systems, you may need to install a newer compiler

### Test Failures
- Run with verbose output: `ctest -C Debug -V`
- Run individual test with diagnostics: `./test_name --verbosity trace`

## Code Architecture Notes

### Node Execution Flow
1. **Initialization:** `run_fft_division_node()` parses JSON parameters
2. **Lane Planning:** `fftdiv_prepare_lane_plan()` determines active lanes
3. **Stage Execution:** 5 pipeline stages process data
   - Stage 1: PCM input → FFT
   - Stage 2: Working tensor operations
   - Stage 3: Spectral processing
   - Stage 4: Spectral output (mailbox)
   - Stage 5: ISTFT → PCM output (mailbox)

### Key Concepts
- **Slots:** Independent processing lanes (typically 1 per channel)
- **Lanes:** Logic bindings between slots and tensor indices
- **io_mode:** Controls whether node outputs PCM, spectral data, or both
  - `"pcm"` (0): FFT→ISTFT roundtrip, PCM mailbox output
  - `"spectral"` (1): FFT only, spectral mailbox output
  - `"both"` (2): FFT→ISTFT, both mailboxes active

## Recent Changes

The recent fix addresses an issue where `io_mode="spectral"` was not properly honored:
- Added `io_mode` field to fftdiv state structure (amp_kernels.c line ~1958)
- Parse `io_mode` from JSON parameters during initialization (fft_division_nodes.inc line ~1472)
- Skip inverse FFT handle creation when `io_mode=1` (spectral) (amp_kernels.c line ~3596)
- Update lane planning to disable PCM output in spectral mode (fft_division_nodes.inc line ~1311)
- Suppress ANOMALY log when zero PCM is expected in spectral mode (fft_division_nodes.inc line ~3032)
- Stage 5 (ISTFT) is now skipped when no inverse handle exists (already checked at line ~2931)

### Problem Statement

The test `test_fft_division_node` was failing with repeated log messages:
```
[STAGE5-PCM-DISPATCH] pcm_written=0 frames_to_push=0 ANOMALY
```

This occurred because:
1. The `io_mode="spectral"` parameter was being ignored
2. Inverse FFT handles were created even in spectral-only mode
3. Stage5 attempted to produce PCM output even when none was expected
4. Zero PCM output was flagged as anomalous in all modes

### Solution

The fix implements proper `io_mode` handling through defense-in-depth:
- **Prevention at creation**: Don't create inverse FFT handle in spectral mode
- **Prevention at enqueue**: Don't queue spectra for ISTFT when `enable_pcm_out=false`
- **Prevention at execution**: Stage5 checks for null inverse handle
- **Proper logging**: Suppress false ANOMALY messages in spectral mode

## Contact & Resources

- Repository: https://github.com/sangderenard/amp
- Issues: File via GitHub Issues
- Related: fftfree library at https://github.com/sangderenard/fftfree
