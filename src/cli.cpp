#include "cli.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cxxopts.hpp"

namespace fs = std::filesystem;

namespace sdf2opmd
{
std::vector<std::string>
splitList(std::string const & list, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, sep))
    {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

Cli
Cli::parse(int argc, char * * argv)
{
    Cli cli;

    cxxopts::Options opts("sdf2opmd",
        "Convert EPOCH/SDF snapshots to openPMD (ADIOS2 BP5).\n"
        "Reads SDF files with the SDF_C library via collective MPI-IO and\n"
        "writes one openPMD iteration file per input file.");

    // clang-format off
    opts.add_options()
        ("h,help", "print help")
        ("p,dir", "directory with .sdf files",
            cxxopts::value<std::string>(cli.dir))
        ("f,files", "colon-separated .sdf file subset",
            cxxopts::value<std::string>(cli.files))
        ("o,outdir", "output directory (default: --dir)",
            cxxopts::value<std::string>(cli.outdir))
        ("n,name", "iteration file pattern (default simData_%04T.bp)",
            cxxopts::value<std::string>(cli.name))
        ("m,fields", "colon-separated SDF field ids (default: all)",
            cxxopts::value<std::string>())
        ("s,species", "colon-separated species names (default: all)",
            cxxopts::value<std::string>())
        ("engine", "adios2 engine JSON config fragment (advanced)",
            cxxopts::value<std::string>(cli.engine))
        ("author", "openPMD author attribute",
            cxxopts::value<std::string>(cli.author))
        ("machine", "openPMD machine attribute",
            cxxopts::value<std::string>(cli.machine))
        ("info", "print SDF block information and exit")
        ("quiet", "reduce output");
    // clang-format on

    auto result = opts.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << opts.help() << std::endl;
        std::exit(0);
    }
    cli.info = result.count("info") > 0;
    cli.quiet = result.count("quiet") > 0;

    if (result.count("fields"))
        cli.fields = splitList(result["fields"].as<std::string>());
    if (result.count("species"))
        cli.species = splitList(result["species"].as<std::string>());

    if (cli.dir.empty() && cli.files.empty())
        throw std::runtime_error("nothing to do: give --dir and/or --files");

    // accept both %T and padded %0NdT forms
    bool hasPattern = false;
    for (size_t i = 0; i + 1 < cli.name.size() && !hasPattern; ++i)
    {
        if (cli.name[i] != '%')
            continue;
        size_t j = i + 1;
        while (j + 1 < cli.name.size() && std::isdigit(unsigned(cli.name[j])))
            ++j;
        hasPattern = cli.name[j] == 'T';
    }
    if (!hasPattern)
        throw std::runtime_error("--name must contain a %T iteration pattern");

    return cli;
}

std::vector<std::string>
Cli::resolveFiles() const
{
    std::vector<std::string> out;

    if (!files.empty())
    {
        for (auto & f : splitList(files))
        {
            fs::path p = f;
            if (p.is_relative() && !dir.empty())
                p = fs::path(dir) / f;
            out.push_back(p.string());
        }
    }
    else if (!dir.empty())
    {
        std::error_code ec;
        if (!fs::is_directory(dir, ec))
            throw std::runtime_error("not a directory: " + dir);
        for (auto const & entry : fs::directory_iterator(dir))
        {
            auto const path = entry.path();
            if (path.extension() == ".sdf")
                out.push_back(path.string());
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());

    if (out.empty())
        throw std::runtime_error("no .sdf files found");

    return out;
}

} // namespace sdf2opmd
