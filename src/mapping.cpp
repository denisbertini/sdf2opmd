#include "mapping.hpp"

#include <algorithm>
#include <cctype>

#include <openPMD/openPMD.hpp>

namespace sdf2opmd
{
std::string
sanitize(std::string name)
{
    for (auto & c : name)
    {
        unsigned char const u = c;
        bool const ok = std::isalnum(u) || c == '_' || c == '-' || c == '.';
        if (!ok)
            c = '_';
    }
    if (!name.empty() && std::isdigit(static_cast<unsigned char>(name[0])))
        name.insert(name.begin(), '_');
    return name;
}

FieldTarget
fieldTarget(std::string const & id)
{
    static std::map<std::string, std::pair<char const *, char const *>> const
        table = {
            {"ex", {"E", "x"}}, {"ey", {"E", "y"}}, {"ez", {"E", "z"}},
            {"bx", {"B", "x"}}, {"by", {"B", "y"}}, {"bz", {"B", "z"}},
            {"jx", {"J", "x"}}, {"jy", {"J", "y"}}, {"jz", {"J", "z"}},
        };

    auto it = table.find(id);
    if (it != table.end())
        return {it->second.first, it->second.second};

    return {sanitize(id), ""};
}

bool
fieldUnitDimension(std::string const & name,
    std::map<openPMD::UnitDimension, double> & out)
{
    using U = openPMD::UnitDimension;
    if (name == "E")
    {
        out = {{U::L, 1}, {U::M, 1}, {U::T, -3}, {U::I, -1}};
        return true;
    }
    if (name == "B")
    {
        out = {{U::M, 1}, {U::T, -2}, {U::I, -1}};
        return true;
    }
    if (name == "J")
    {
        out = {{U::L, -2}, {U::I, 1}};
        return true;
    }
    if (name.rfind("rho", 0) == 0) // rho, rho_total, ...
    {
        out = {{U::L, -3}, {U::I, 1}, {U::T, 1}};
        return true;
    }
    return false;
}

PartTarget
particleTarget(std::string const & varId, std::string const & species)
{
    std::string base = varId;
    std::string const suffix = "_" + species;
    if (base.size() > suffix.size()
        && base.compare(base.size() - suffix.size(), suffix.size(), suffix)
            == 0)
    {
        base.erase(base.size() - suffix.size());
    }

    if (base == "x" || base == "y" || base == "z")
        return {"position", base};
    if (base == "px" || base == "py" || base == "pz")
        return {"momentum", std::string(1, base[1])};
    if (base == "vx" || base == "vy" || base == "vz")
        return {"velocity", std::string(1, base[1])};
    if (base == "w")
        return {"weighting", ""};

    return {sanitize(base), ""};
}

bool
particleUnitDimension(std::string const & record,
    std::map<openPMD::UnitDimension, double> & out)
{
    using U = openPMD::UnitDimension;
    if (record == "position")
    {
        out = {{U::L, 1}};
        return true;
    }
    if (record == "momentum")
    {
        out = {{U::L, 1}, {U::M, 1}, {U::T, -1}};
        return true;
    }
    if (record == "velocity")
    {
        out = {{U::L, 1}, {U::T, -1}};
        return true;
    }
    if (record == "mass")
    {
        out = {{U::M, 1}};
        return true;
    }
    if (record == "charge")
    {
        out = {{U::T, 1}, {U::I, 1}};
        return true;
    }
    if (record == "weighting")
    {
        out.clear();
        return true;
    }
    return false;
}

bool
staggerToPosition(int stagger, int ndims, std::vector<double> & position)
{
    int const all = (1 << ndims) - 1;
    if (ndims < 1 || ndims > 8 || stagger < 0 || stagger > all)
        return false;
    if (stagger == 0) // SDF_STAGGER_CELL_CENTRE: half a cell off everywhere
    {
        position.assign(ndims, 0.5);
        return true;
    }
    if (stagger == all) // node centred: exactly on the mesh
    {
        position.assign(ndims, 0.0);
        return true;
    }
    // Bit set == half-cell offset on that axis (face/edge centres). This
    // matches the SDF bit convention (see enum sdf_stagger in sdf.h).
    position.assign(size_t(ndims), 0.0);
    for (int d = 0; d < ndims; ++d)
        if (stagger & (1 << d))
            position[size_t(d)] = 0.5;
    return true;
}

} // namespace sdf2opmd
