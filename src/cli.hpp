#pragma once
//
// Command line configuration.
//
#include <string>
#include <vector>

namespace sdf2opmd
{
struct Cli
{
    std::string dir;            // directory containing .sdf files
    std::string files;          // colon-separated list of .sdf files (subset)
    std::string outdir;         // output directory (default: dir)
    std::string name = "simData_%04T.bp"; // iteration file pattern
    std::vector<std::string> fields;  // field filter (empty = all)
    std::vector<std::string> species; // species filter (empty = all)
    std::string author = "sdf2opmd";
    std::string machine = "";
    std::string engine;         // raw adios2 JSON config fragment (optional)
    bool info = false;          // print SDF block table, don't convert
    bool quiet = false;

    // Parses argv. Returns std::nullopt when the program should exit without
    // doing work (--help was shown or a parse error occurred).
    static Cli
    parse(int argc, char * * argv);

    // Resolves --dir/--files to a sorted, de-duplicated list of input files.
    std::vector<std::string>
    resolveFiles() const;
};

std::vector<std::string>
splitList(std::string const & csv, char sep = ':');

} // namespace sdf2opmd
