#pragma once
//
// EPOCH/SDF naming conventions -> openPMD object model mapping.
//
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <openPMD/UnitDimension.hpp>

namespace sdf2opmd
{
// Replace characters that are unsafe in openPMD/ADIOS2 object names.
std::string
sanitize(std::string name);

struct FieldTarget
{
    std::string mesh;      // openPMD mesh (record group) name
    std::string component; // "" means scalar
};

// ex/ey/ez -> E/x.., bx.. -> B/x.., jx.. -> J/x..; anything else becomes a
// scalar mesh named after the (sanitized) SDF id.
FieldTarget
fieldTarget(std::string const & sdfId);

// Unit dimensions for well-known field meshes (openPMD standard).
// Returns false if no standard dimension applies.
bool
fieldUnitDimension(std::string const & meshName,
    std::map<openPMD::UnitDimension, double> & out);

struct PartTarget
{
    std::string record;
    std::string component; // "" means scalar
};

// Maps an SDF point-variable id (e.g. "px_el") for a species to an openPMD
// record/component: x,y,z -> position; px,py,pz -> momentum; w -> weighting;
// anything else keeps its own name as a custom scalar record.
PartTarget
particleTarget(std::string const & varId, std::string const & species);

bool
particleUnitDimension(std::string const & recordName,
    std::map<openPMD::UnitDimension, double> & out);

// openPMD mesh position (grid offset relative to nodes) from an SDF stagger
// bitmask. Returns false when the stagger is not representable.
bool
staggerToPosition(int stagger, int ndims, std::vector<double> & position);

} // namespace sdf2opmd
