// sdfdump: raw bytes of one SDF block, for validators that outgrew their
// reader (e.g. python 'sdfr' overflows int32 on blocks > 2 GiB).
//
//   sdfdump SDF_FILE BLOCK_ID OUTFILE   -> stdout: "<datatype> <nbytes>"
//
// Single rank; absolute-offset MPI-IO reads in < 2 GiB chunks. Uses the
// exact same SDF_C metadata + MPI-IO path as the converter, so what it
// prints is what the converter saw.
#include "sdf_file.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace sdf2opmd;

int
main(int argc, char ** argv)
{
    MpiScope const mpi(argc, &argv);
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: sdfdump SDF_FILE BLOCK_ID OUTFILE\n");
        return 2;
    }
    CommSize const cs(MPI_COMM_WORLD);
    if (cs.size != 1)
    {
        std::fprintf(stderr, "sdfdump must run on a single rank\n");
        return 2;
    }

    SdfFile file = SdfFile::open(MPI_COMM_WORLD, argv[1]);
    sdf_block_t * b = nullptr;
    for (auto * p : file.blocks())
        if (p->id && std::strcmp(p->id, argv[2]) == 0)
            b = p;
    if (!b)
    {
        std::fprintf(stderr, "no block named '%s' in %s\n", argv[2], argv[1]);
        return 2;
    }
    // SDF datatypes: 0 null | 1 i4 | 2 i8 | 3 r4 | 4 r8 | 5 r16 | 6 char
    //                | 7 logical(1B)
    static int const ts[] = {0, 4, 8, 4, 8, 16, 1, 1};
    if (b->datatype < 1 || b->datatype > 7 || b->data_location < 0
        || b->data_length <= 0)
    {
        std::fprintf(stderr, "block '%s': unsupported datatype/extent\n",
            argv[2]);
        return 2;
    }
    (void)ts; // datatype id is echoed; numpy dtype mapping lives client-side

    std::FILE * out = std::fopen(argv[3], "wb");
    if (!out)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[3]);
        return 2;
    }

    int64_t off = 0;
    int64_t const chunk = int64_t(512) << 20;
    std::vector<char> buf(size_t(chunk), 0);
    while (off < b->data_length)
    {
        int64_t const n = std::min(chunk, b->data_length - off);
        MPI_Status st;
        mpiCheck(MPI_File_read_at(file.fh(),
                       MPI_Offset(b->data_location + off), buf.data(),
                       int(n), MPI_BYTE, &st),
            "sdfdump read_at");
        int got = 0;
        mpiCheck(MPI_Get_count(&st, MPI_BYTE, &got), "sdfdump count");
        if (int64_t(got) != n)
        {
            std::fprintf(stderr, "short read at %lld\n", (long long)off);
            return 1;
        }
        std::fwrite(buf.data(), 1, size_t(n), out);
        off += n;
    }
    std::fclose(out);
    std::printf("%d %lld\n", b->datatype, (long long)b->data_length);
    return 0;
}
