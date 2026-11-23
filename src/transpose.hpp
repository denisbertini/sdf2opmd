#pragma once
//
// Cache-blocked transposes from Fortran (first dim fastest) to C (last dim
// fastest) memory order.
//
// Note: an MPI-derived-datatype "free transpose" is impossible here: MPI-IO
// requires filetype displacements to be monotonically non-decreasing
// (ROMIO's iobadoverlap check), and a reversed traversal of a Fortran-ordered
// file is by construction non-monotonic. So the read is done in monotonic
// Fortran order and the reordering happens here, in parallel per rank.
//
#include <algorithm>
#include <cstring>

namespace sdf2opmd
{

template <typename T>
inline void
transpose2d(const T * src, T * dst, int64_t nx, int64_t ny)
{
    // src: Fortran order [i + nx*j]; dst: C order [i*ny + j]
    constexpr int64_t TB = 32;
    for (int64_t i0 = 0; i0 < nx; i0 += TB)
    {
        for (int64_t j0 = 0; j0 < ny; j0 += TB)
        {
            const int64_t i1 = std::min(i0 + TB, nx);
            const int64_t j1 = std::min(j0 + TB, ny);
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t i = i0; i < i1; ++i)
                    dst[i * ny + j] = src[i + nx * j];
            }
        }
    }
}

template <typename T>
inline void
transpose3d(const T * src, T * dst, int64_t nx, int64_t ny, int64_t nz)
{
    // src: Fortran order [i + nx*(j + ny*k)]
    // dst: C order      [i*ny*nz + j*nz + k]
    constexpr int64_t TB = 16;
    for (int64_t i0 = 0; i0 < nx; i0 += TB)
    {
        for (int64_t j0 = 0; j0 < ny; j0 += TB)
        {
            for (int64_t k0 = 0; k0 < nz; k0 += TB)
            {
                const int64_t i1 = std::min(i0 + TB, nx);
                const int64_t j1 = std::min(j0 + TB, ny);
                const int64_t k1 = std::min(k0 + TB, nz);
                for (int64_t i = i0; i < i1; ++i)
                {
                    for (int64_t j = j0; j < j1; ++j)
                    {
                        const T * s = src + i + nx * j;
                        T * d = dst + i * ny * nz + j * nz;
                        for (int64_t k = k0; k < k1; ++k)
                            d[k] = s[nx * ny * k];
                    }
                }
            }
        }
    }
}

template <typename T>
inline void
transpose4d(const T * src, T * dst, const int64_t dims[4])
{
    const int64_t nx = dims[0];
    const int64_t ny = dims[1];
    const int64_t nz = dims[2];
    const int64_t nw = dims[3];
    for (int64_t l = 0; l < nw; ++l)
    {
        const int64_t sOff = nx * ny * nz * l;
        const int64_t dOff = ny * nz * nw * l;
        transpose3d(src + sOff, dst + dOff, nx, ny, nz);
    }
}

// dims are in SDF (Fortran) order: dims[0] fastest in file == x.
template <typename T>
inline void
transposeFtoC(const T * src, T * dst, const int64_t * dims, int ndims)
{
    switch (ndims)
    {
    case 1:
        std::memcpy(dst, src, sizeof(T) * dims[0]);
        break;
    case 2: transpose2d(src, dst, dims[0], dims[1]); break;
    case 3: transpose3d(src, dst, dims[0], dims[1], dims[2]); break;
    case 4: transpose4d(src, dst, dims); break;
    default: break;
    }
}

} // namespace sdf2opmd
