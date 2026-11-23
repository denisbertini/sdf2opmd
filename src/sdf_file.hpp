#pragma once
//
// RAII wrapper around the SDF C library handle plus the raw MPI-IO read
// primitives used by the converter.
//
// The read helpers below deliberately bypass SDF_C's own data-reading
// routines and build their own MPI-IO views instead:
//
//  * readFieldSlab() reads a rank slab through a Fortran-ordered subarray
//    view (contiguous per-row accesses, monotonic displacements), then
//    reorders to C order with a cache-blocked per-rank transpose. A
//    "free transpose" via a reversed C-order filetype is impossible:
//    MPI-IO requires filetype displacements to be monotonically
//    non-decreasing (ROMIO's iobadoverlap check).
//
//  * readPointSlab() reads a contiguous particle range.
//
//  * readWhole() reads one small block on the master rank and broadcasts.
//
// All three helpers are collective over the file's communicator.
//
#include <mpi.h>

#include <sdf.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpi.hpp"
#include "transpose.hpp"

namespace sdf2opmd
{
template <typename T>
struct MpiBase
{
    static MPI_Datatype
    value()
    {
        if constexpr (std::is_same_v<T, double>)
            return MPI_DOUBLE;
        else if constexpr (std::is_same_v<T, float>)
            return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, int32_t>)
            return MPI_INT;
        else if constexpr (std::is_same_v<T, int64_t>)
            return MPI_LONG_LONG;
        else
            static_assert(sizeof(T) == 0, "no MPI base type");
    }
};

inline int
checkedInt(int64_t v)
{
    if (v > int64_t(2147483647LL))
        throw std::runtime_error("count exceeds int range");
    return int(v);
}

// Debug aid: SDF2OPMD_FAKE_READS=1 replaces data reads with synthetic fills
// (metadata parsing and openPMD writes stay real) to isolate I/O bugs.
// Fine-grained switches: SDF2OPMD_FAKE_FIELDS / _POINT / _WHOLE.
inline bool
fakeEnv(const char * name)
{
    return getenv("SDF2OPMD_FAKE_READS") != nullptr
        || getenv(name) != nullptr;
}

class SdfFile
{
public:
    static SdfFile
    open(MPI_Comm comm, std::string const & path)
    {
        sdf_file_t * h = sdf_open(path.c_str(), comm, SDF_READ, 0);
        if (!h)
            throw std::runtime_error("cannot open SDF file: " + path);
        h->print = 0; // silence block dumps
        SdfFile f(h);
        if (sdf_read_blocklist(f.h_) != 0)
            throw std::runtime_error("cannot read blocklist: " + path);
        return f;
    }

    SdfFile(SdfFile && other) noexcept : h_(other.h_)
    {
        other.h_ = nullptr;
    }

    SdfFile &
    operator=(SdfFile && other) noexcept
    {
        if (this != &other)
        {
            reset();
            h_ = other.h_;
            other.h_ = nullptr;
        }
        return *this;
    }

    SdfFile(SdfFile const &) = delete;
    SdfFile & operator=(SdfFile const &) = delete;

    ~SdfFile()
    {
        reset();
    }

    sdf_file_t *
    raw()
    {
        return h_;
    }

    int32_t
    step() const
    {
        return h_->step;
    }

    double
    time() const
    {
        return h_->time;
    }

    std::string
    codeName() const
    {
        return h_->code_name ? h_->code_name : "";
    }

    MPI_File
    fh() const
    {
        return h_->filehandle;
    }

    MPI_Comm
    comm() const
    {
        return h_->comm;
    }

    int
    rank() const
    {
        return h_->rank;
    }

    int
    master() const
    {
        return h_->rank_master;
    }

    // Blocks in file order; scrubbed/empty entries are skipped.
    std::vector<sdf_block_t *>
    blocks()
    {
        std::vector<sdf_block_t *> out;
        sdf_block_t * b = h_->blocklist;
        for (int i = 0; i < h_->nblocks && b; ++i)
        {
            auto * next = b->next;
            if (b->blocktype != SDF_BLOCKTYPE_SCRUBBED)
                out.push_back(b);
            b = next;
        }
        return out;
    }

private:
    explicit SdfFile(sdf_file_t * h) : h_(h)
    {}

    void
    reset()
    {
        if (h_)
        {
            sdf_close(h_); // closes file, frees blocks and the handle
            h_ = nullptr;
        }
    }

    sdf_file_t * h_ = nullptr;
};

/* Collective. Every rank must call this, even ranks with counts[] == 0
 * (they simply read nothing but still join the view switch).
 *
 * Requires b.datatype to match sizeof(T). starts/counts refer to the
 * per-dimension global Fortran indices (dim 0 = x = fastest in file).
 *
 * The file view is a Fortran-ordered subarray (monotonic displacements, the
 * fastest path in ROMIO); the C-order buffer that openPMD expects is
 * produced afterwards by a blocked transpose on each rank.
 */
template <typename T>
inline std::unique_ptr<T[]>
readFieldSlab(SdfFile & f, sdf_block_t const & b, int64_t const starts[4],
    int64_t const counts[4])
{
    int const nd = b.ndims;
    int g[SDF_MAXDIMS];
    int l[SDF_MAXDIMS];
    int s[SDF_MAXDIMS];
    int64_t n = 1;
    for (int d = 0; d < nd; ++d)
    {
        g[d] = checkedInt(b.dims[d]);
        l[d] = checkedInt(counts[d]);
        s[d] = checkedInt(starts[d]);
        n *= counts[d];
    }

    if (fakeEnv("SDF2OPMD_FAKE_FIELDS"))
    {
        std::unique_ptr<T[]> cbuf(new T[std::max<int64_t>(n, 1)]());
        for (int64_t i = 0; i < n; ++i)
            cbuf[i] = T(i);
        return cbuf;
    }

    MPI_Datatype type = MPI_DATATYPE_NULL;
    mpiCheck(MPI_Type_create_subarray(nd, g, l, s, MPI_ORDER_FORTRAN,
                       MpiBase<T>::value(), &type),
        "MPI_Type_create_subarray");
    mpiCheck(MPI_Type_commit(&type), "MPI_Type_commit");

    std::unique_ptr<T[]> fbuf(new T[std::max<int64_t>(n, 1)]());

    mpiCheck(MPI_File_set_view(f.fh(), MPI_Offset(b.data_location),
                       MpiBase<T>::value(), type, "native", MPI_INFO_NULL),
        "MPI_File_set_view(field)");
    // With a derived filetype view, the view exposes exactly the mapped
    // region elements in file order; the buffer datatype must therefore be
    // the plain base type with count = number of region elements. (Passing
    // the subarray type again here would scatter the view stream at map
    // positions and write far past the buffer.)
    mpiCheck(MPI_File_read_all(
                 f.fh(), fbuf.get(), checkedInt(n), MpiBase<T>::value(),
                 MPI_STATUS_IGNORE),
        "MPI_File_read_all(field)");
    mpiCheck(MPI_File_set_view(f.fh(), 0, MPI_BYTE, MPI_BYTE, "native",
                       MPI_INFO_NULL),
        "MPI_File_set_view(reset)");
    MPI_Type_free(&type);

    std::unique_ptr<T[]> cbuf(new T[std::max<int64_t>(n, 1)]());
    transposeFtoC(fbuf.get(), cbuf.get(), counts, nd);
    return cbuf;
}

/* Collective contiguous slab read for point (particle) data. */
template <typename T>
inline std::unique_ptr<T[]>
readPointSlab(SdfFile & f, sdf_block_t const & b, int64_t start,
    int64_t count)
{
    auto const sz = static_cast<int64_t>(sizeof(T));

    if (fakeEnv("SDF2OPMD_FAKE_POINT"))
    {
        std::unique_ptr<T[]> buf(new T[std::max<int64_t>(count, 1)]());
        for (int64_t i = 0; i < count; ++i)
            buf[i] = T(i);
        return buf;
    }

    std::unique_ptr<T[]> buf(new T[std::max<int64_t>(count, 1)]());

    if (count * sz > int64_t(2147483647LL))
        throw std::runtime_error("point slab too large for one read");

    mpiCheck(MPI_File_set_view(f.fh(), MPI_Offset(b.data_location + start * sz),
                       MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL),
        "MPI_File_set_view(point)");
    mpiCheck(MPI_File_read_all(f.fh(), buf.get(), int(count * sz), MPI_BYTE,
                       MPI_STATUS_IGNORE),
        "MPI_File_read_all(point)");
    mpiCheck(MPI_File_set_view(f.fh(), 0, MPI_BYTE, MPI_BYTE, "native",
                       MPI_INFO_NULL),
        "MPI_File_set_view(reset)");

    return buf;
}

/* Collective read of a whole small block: rank `master` does one positional
 * read, then broadcasts. Intended for mesh axes etc. (not field data). */
template <typename T>
inline std::vector<T>
readWhole(SdfFile & f, sdf_block_t const & b)
{
    int64_t const nbytes = b.data_length;
    if (nbytes <= 0 || nbytes % int64_t(sizeof(T)) != 0)
        throw std::runtime_error("unexpected data_length for block "
            + std::string(b.id ? b.id : "?"));
    if (nbytes > (int64_t(1) << 31))
        throw std::runtime_error("block too large for readWhole: "
            + std::string(b.id ? b.id : "?"));

    std::vector<T> buf(std::max<int64_t>(nbytes / int64_t(sizeof(T)), 1));

    if (getenv("SDF2OPMD_DEBUG_IO"))
        fprintf(stderr, "[readWhole r%d] id=%s loc=%lld nbytes=%lld vec=%zu\n",
            f.rank(), b.id ? b.id : "?", (long long)b.data_location,
            (long long)nbytes, buf.size());

    if (fakeEnv("SDF2OPMD_FAKE_WHOLE"))
    {
        for (size_t i = 0; i < buf.size(); ++i)
            buf[i] = T(0.1 * double(i));
        return buf;
    }

    mpiCheck(MPI_Barrier(f.comm()), "MPI_Barrier(readWhole)");
    if (f.rank() == f.master())
    {
        MPI_Status st;
        mpiCheck(MPI_File_read_at(f.fh(), MPI_Offset(b.data_location),
                       buf.data(), int(nbytes), MPI_BYTE, &st),
            "MPI_File_read_at");
    }
    mpiCheck(MPI_Bcast(buf.data(), int(nbytes), MPI_BYTE, f.master(),
                       f.comm()),
        "MPI_Bcast(readWhole)");
    mpiCheck(MPI_Barrier(f.comm()), "MPI_Barrier(readWhole)");

    return buf;
}

} // namespace sdf2opmd
