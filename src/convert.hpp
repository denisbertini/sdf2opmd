#pragma once
//
// Per-file conversion: SDF -> openPMD iteration.
//
#include <openPMD/openPMD.hpp>

#include <chrono>

#include "blockmap.hpp"
#include "cli.hpp"
#include "mapping.hpp"
#include "mpi.hpp"
#include "sdf_file.hpp"

namespace sdf2opmd
{
inline double
now()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch())
        .count();
}

struct Timings
{
    double meta = 0;
    double read = 0;
    double flush = 0;
    double total = 0;
    int files = 0;
    double peakRssMiB = 0; // process high-water mark (per rank, max over files)

    void
    print(MPI_Comm comm) const;
};

openPMD::Series
makeSeries(Cli const & cli, MPI_Comm comm);

// Converts one open SDF file into one openPMD iteration. Collective.
void
convertFile(openPMD::Series & series, SdfFile & file,
    std::string const & path, Cli const & cli, Timings & timings);

// Prints a human-readable summary of the SDF contents (rank 0 only).
void
printInfo(SdfFile & file, std::string const & path, BlockMap const & map);

// Dispatch f.template operator()<T>() on the SDF datatype of a block.
template <typename F>
void
withDtype(int32_t sdfDatatype, F && f)
{
    switch (sdfDatatype)
    {
    case SDF_DATATYPE_REAL8:
        std::forward<F>(f).template operator()<double>();
        break;
    case SDF_DATATYPE_REAL4:
        std::forward<F>(f).template operator()<float>();
        break;
    case SDF_DATATYPE_INTEGER4:
        std::forward<F>(f).template operator()<int32_t>();
        break;
    case SDF_DATATYPE_INTEGER8:
        std::forward<F>(f).template operator()<int64_t>();
        break;
    default:
        throw std::runtime_error("unsupported SDF datatype");
    }
}

// Even 1D split of a global extent: rank-local [start, start+count).
struct Slab
{
    int64_t start = 0;
    int64_t count = 0;
};

inline Slab
evenSplit(int64_t global, int size, int rank)
{
    Slab s;
    if (global <= 0 || rank >= global)
        return s;
    int64_t const base = global / size;
    int64_t const rem = global % size;
    if (rank < rem)
    {
        s.count = base + 1;
        s.start = int64_t(rank) * (base + 1);
    }
    else
    {
        s.count = base;
        s.start = rem * (base + 1) + int64_t(rank - rem) * base;
    }
    return s;
}

} // namespace sdf2opmd
