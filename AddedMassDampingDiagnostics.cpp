// ---------------------------------------------------------------------
// AddedMassDampingDiagnostics.cpp
//
// Purpose:
//   Output per-slice hydrodynamic constraint force acting on the structure
//   for post-processing of added mass and added damping.
//
// Design principle:
//   - IBAMR solver outputs ONLY solver-dependent quantities.
//   - No analytical kinematics or derived coefficients are computed here.
//   - All added-mass and damping formulas are applied in post-processing
//     (e.g. compute_added_mass_damping.m or compute_cylinder_added_mass.m).
//
// Output:
//   time  F_L_0  F_L_1  ...  F_L_{N_s-1}
//
// where F_L_n(t) is the constraint force on slice n in the configured
// direction (force_direction: 0=x, 1=y).
//
// This design matches:
//   - ConstraintIBMethod force evaluation
//   - Fish2d_Drag_force_struct_no_* diagnostics
//   - Theoretical formulation used in the literature
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

// Class declaration for this diagnostic object
#include "AddedMassDampingDiagnostics.h"

// MPI utilities used for rank checks and global reductions
#include <ibtk/IBTK_MPI.h>

// LData stores Lagrangian data (e.g. constraint forces) on each level
#include <ibtk/LData.h>

// LDataManager maps between Lagrangian indices and physical data
#include <ibtk/LDataManager.h>

// LMesh provides access to local Lagrangian nodes on each MPI rank
#include <ibtk/LMesh.h>

// LNode represents an individual Lagrangian point
#include <ibtk/LNode.h>

// PatchLevel gives access to the AMR hierarchy level information
#include <PatchLevel.h>

// File I/O for writing diagnostic output
#include <fstream>

// std::floor for slice index computation.
#include <cmath>

// I/O manipulators for output precision control
#include <iomanip>

// String parsing utilities (used for reading .vertex file)
#include <sstream>

// IBAMR application-wide namespace imports (Pointer, TBOX_ASSERT, pout, etc.)
#include <ibamr/app_namespaces.h>


/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace IBAMR
{

/////////////////////////////// PUBLIC ///////////////////////////////////////

AddedMassDampingDiagnostics::AddedMassDampingDiagnostics(
    const std::string& object_name,
    Pointer<Database> input_db,
    Pointer<ConstraintIBMethod> ib_method_ops,
    Pointer<PatchHierarchy<NDIM> > patch_hierarchy)
    :
    // Unique name for this diagnostics object (used only for logging/debugging)
    d_object_name(object_name),

    // Pointer to ConstraintIBMethod: provides access to constraint forces
    // (Lagrange multipliers enforcing prescribed kinematics)
    d_ib_method_ops(ib_method_ops),

    // Patch hierarchy: required to determine finest level and Lagrangian indexing
    d_patch_hierarchy(patch_hierarchy),

    // Flag indicating whether the output file was successfully opened (rank 0 only)
    d_file_opened(false)
{
    // ------------------------------------------------------------------
    // Solver-only parameters (NO kinematics, NO analytical quantities)
    // These parameters define geometry, indexing, and physical scaling
    // used directly by the solver.
    // ------------------------------------------------------------------

    // Non-dimensional chord length of the foil / reference length.
    // In IBAMR, the Navier-Stokes equations are solved in non-dimensional form,
    // so L = 1.0 is the natural reference length.
    d_L = input_db->getDoubleWithDefault("L", 1.0);

    // Fluid density (non-dimensional, consistent with RHO in input2d).
    // Required to convert velocity correction into physical force.
    d_rho = input_db->getDoubleWithDefault("rho", 1.0);

    // Number of chordwise slices used to spatially bin constraint forces.
    // This is a numerical quadrature / resolution parameter,
    // NOT a physical parameter.
    d_N_s = input_db->getIntegerWithDefault("N_s", 20);

    // Structure ID handled by this diagnostics object.
    // Allows extension to multi-body simulations.
    d_structure_id = input_db->getIntegerWithDefault("structure_id", 0);

    // Vertex file defining the reference (time-independent) geometry.
    // This is the SAME file used by IBStandardInitializer.
    // No runtime LData ("X" or "X0") is used for geometry.
    d_vertex_filename = input_db->getString("vertex_filename");

    // Force extraction direction: 0=x, 1=y.
    // Default 1 preserves backward compatibility with eel2d (transverse/lift).
    // Set to 0 for bodies oscillating in x-direction (e.g. cylinder).
    d_force_direction = input_db->getIntegerWithDefault("force_direction", 1);

    // Validate force_direction
    if (d_force_direction < 0 || d_force_direction >= NDIM)
    {
        TBOX_ERROR(d_object_name << "::AddedMassDampingDiagnostics():\n"
                   << "  force_direction = " << d_force_direction
                   << " is out of range [0, " << NDIM - 1 << "].\n");
    }

    // ------------------------------------------------------------------
    // Storage for per-slice constraint forces
    // ------------------------------------------------------------------

    // Cached per-slice forces F_L_n(t), filled during integrateHierarchy().
    // Size = number of slices.
    d_cached_F_L_n.resize(d_N_s, 0.0);

    // Indicates whether constraint forces have been captured
    // during the current timestep.
    d_force_captured = false;

    // ------------------------------------------------------------------
    // Slice membership initialization
    // ------------------------------------------------------------------

    // Reads the .vertex file, determines:
    //   - Leading-edge location
    //   - Lagrangian index range for this structure
    //   - Mapping from Lagrangian points -> slice index
    //
    // This uses ONLY reference geometry and is done ONCE at initialization.
    pout << ">>> DEBUG: About to call initializeSliceMembership()...\n" << std::flush;
    initializeSliceMembership();
    pout << ">>> DEBUG: initializeSliceMembership() completed.\n" << std::flush;

    // ------------------------------------------------------------------
    // Output file construction
    // ------------------------------------------------------------------

    // Output directory (must match ConstraintIBMethod output convention)
    std::string output_dirname =
        input_db->getStringWithDefault("output_dirname", "./Fish2dStr");

    // Base filename (again matching IBAMR naming style)
    std::string base_filename =
        input_db->getStringWithDefault("base_filename", "Fish2d");

    // Structure ID appended for multi-body consistency
    std::string struct_id_str = std::to_string(d_structure_id);

    // Final output filename:
    // <base>_slice_force_struct_no_<id>
    d_output_filename =
        output_dirname + "/" +
        base_filename + "_slice_force_struct_no_" + struct_id_str;

    // ------------------------------------------------------------------
    // File opening (MPI rank 0 only)
    // ------------------------------------------------------------------

    // Only rank 0 performs file I/O, consistent with IBAMR practice.
    if (IBTK_MPI::getRank() == 0)
    {
        // Ensure output directory exists before attempting to open the file.
        Utilities::recursiveMkdir(output_dirname);

        // Append mode allows restarts without overwriting data
        d_out_file.open(d_output_filename.c_str(),
                        std::ios::out | std::ios::app);

        if (!d_out_file.is_open())
        {
            TBOX_ERROR(d_object_name << "::AddedMassDampingDiagnostics():\n"
                       << "  Failed to open output file: " << d_output_filename << "\n");
        }
        d_file_opened = true;
        d_out_file << std::setprecision(12);
    }

    // ------------------------------------------------------------------
    // Diagnostic logging (for verification and reproducibility)
    // ------------------------------------------------------------------

    const char* dir_names[] = {"x", "y", "z"};
    pout << "\n============================================================\n";
    pout << "AddedMassDampingDiagnostics initialized\n";
    pout << "  Vertex file: " << d_vertex_filename << "\n";
    pout << "  Num lag pts: " << d_num_lag_pts << "\n";
    pout << "  L = " << d_L << " , N_s = " << d_N_s << "\n";
    pout << "  Force direction: " << d_force_direction
         << " (" << dir_names[d_force_direction] << ")\n";
    pout << "  Leading edge x = " << d_x_leading_edge
         << " (from .vertex file)\n";
    pout << "  Output file: " << d_output_filename << "\n";
    pout << "  Output format: time  F_L_0  F_L_1  ...  F_L_{N_s-1}\n";
    pout << "============================================================\n\n";

    return;
}


AddedMassDampingDiagnostics::~AddedMassDampingDiagnostics()
{
    // Only rank 0 opens the output file, so only rank 0 may need to close it.
    // The check d_file_opened ensures we do not attempt to close an unopened file.
    if (d_file_opened && d_out_file.is_open())
    {
        // Close the per-slice constraint force output file.
        // This safely flushes all buffered data to disk.
        d_out_file.close();
    }

    // No other cleanup is required:
    // - All IBAMR-managed objects (LData, hierarchy, meshes) are owned elsewhere
    // - STL containers clean up automatically
    return;
}


void
AddedMassDampingDiagnostics::computeAndWriteDiagnostics(double current_time)
{
    // This function writes solver-only diagnostic output for the current timestep.
    // It must be called AFTER the constraint force has been captured via the
    // integrateHierarchy callback.

    // Safety check: ensure that constraint forces were captured for this timestep.
    // If this fails, it indicates a mismatch in the callback / output order.
    // Use TBOX_ERROR (unconditional) instead of TBOX_ASSERT (debug-only).
    if (!d_force_captured)
    {
        TBOX_ERROR(d_object_name << "::computeAndWriteDiagnostics():\n"
                   << "  Constraint force was not captured for this timestep.\n"
                   << "  Ensure integrateHierarchyCallbackFcn is registered.\n");
    }

    // Only MPI rank 0 performs file I/O to avoid duplicate output in parallel runs.
    if (IBTK_MPI::getRank() == 0 && d_file_opened)
    {
        // Write the current simulation time as the first column.
        d_out_file << current_time;

        // Write the per-slice constraint force in the configured direction.
        // Each entry F_L_n corresponds to the total constraint force acting on
        // slice n of the structure at this time.
        for (int n = 0; n < d_N_s; ++n)
        {
            d_out_file << '\t' << d_cached_F_L_n[n];
        }

        // End the current timestep record.
        d_out_file << std::endl;
    }

    // Reset the capture flag so that each timestep requires a fresh force capture.
    // This prevents accidental reuse of forces from a previous timestep.
    d_force_captured = false;

    return;
}


/////////////////////////////// PRIVATE //////////////////////////////////////

void
AddedMassDampingDiagnostics::initializeSliceMembership()
{
    // ------------------------------------------------------------------
    // This routine assigns each Lagrangian point of the structure to a
    // chordwise slice based on its *reference* x-coordinate.
    //
    // IMPORTANT:
    // - The reference geometry is read from the .vertex file.
    // - Slice membership is therefore fixed in time.
    // - No runtime-deformed coordinates ("X") are used here.
    //
    // This is consistent with:
    //   - ConstraintIBMethod design
    //   - Non-dimensional formulation
    //   - Post-processing based added-mass/damping theory
    // ------------------------------------------------------------------

    // Open the .vertex file that defines the reference (undeformed)
    // Lagrangian geometry used by IBStandardInitializer.
    std::ifstream vertex_file(d_vertex_filename.c_str());
    TBOX_ASSERT(vertex_file.is_open());

    // ------------------------------------------------------------------
    // First line of .vertex file contains the total number of Lagrangian
    // points across all structures.
    // ------------------------------------------------------------------
    int num_pts_in_file = 0;
    {
        std::string first_line;
        std::getline(vertex_file, first_line);
        std::istringstream iss(first_line);
        iss >> num_pts_in_file;
    }
    TBOX_ASSERT(num_pts_in_file > 0);

    // ------------------------------------------------------------------
    // Read reference coordinates from the .vertex file.
    // Only x-coordinates are required for chordwise slicing.
    //
    // NOTE:
    // - File order matches Lagrangian index ordering used by IBAMR.
    // - This guarantees consistent mapping between file indices and
    //   internal Lagrangian indices.
    // ------------------------------------------------------------------
    std::vector<double> all_ref_x(num_pts_in_file, 0.0);
    for (int i = 0; i < num_pts_in_file; ++i)
    {
        // Read NDIM coordinates per line (works for both 2D and 3D).
        double coords[NDIM];
        for (int d = 0; d < NDIM; ++d)
            vertex_file >> coords[d];
        all_ref_x[i] = coords[0];
    }
    vertex_file.close();

    // ------------------------------------------------------------------
    // Query IBAMR for the Lagrangian index range corresponding to the
    // specified structure_id.
    //
    // This allows:
    // - Multi-structure simulations
    // - Clean extraction of per-structure data
    // ------------------------------------------------------------------
    IBTK::LDataManager* l_data_manager = d_ib_method_ops->getLDataManager();
    const int finest_ln = d_patch_hierarchy->getFinestLevelNumber();

    std::pair<int, int> lag_idx_range =
        l_data_manager->getLagrangianStructureIndexRange(d_structure_id, finest_ln);

    const int struct_start = lag_idx_range.first;
    const int struct_end   = lag_idx_range.second;

    // Number of Lagrangian points belonging to this structure
    d_num_lag_pts = struct_end - struct_start;

    // Sanity check: structure must have points and indices must be in bounds
    if (d_num_lag_pts <= 0)
    {
        TBOX_ERROR(d_object_name << "::initializeSliceMembership():\n"
                   << "  Structure " << d_structure_id << " has no Lagrangian points.\n"
                   << "  Index range: [" << struct_start << ", " << struct_end << ")\n");
    }
    TBOX_ASSERT(struct_start >= 0 && struct_end <= num_pts_in_file);

    // ------------------------------------------------------------------
    // Extract reference x-coordinates for this structure only.
    // These coordinates are stored using structure-local indexing.
    // ------------------------------------------------------------------
    d_X_ref_x.resize(d_num_lag_pts, 0.0);
    for (int i = 0; i < d_num_lag_pts; ++i)
    {
        d_X_ref_x[i] = all_ref_x[struct_start + i];
    }

    // ------------------------------------------------------------------
    // Determine the leading-edge x-location from reference geometry.
    //
    // This defines x = 0 for chordwise normalization.
    // ------------------------------------------------------------------
    d_x_leading_edge = d_X_ref_x[0];
    for (int i = 1; i < d_num_lag_pts; ++i)
    {
        if (d_X_ref_x[i] < d_x_leading_edge)
            d_x_leading_edge = d_X_ref_x[i];
    }

    // ------------------------------------------------------------------
    // Assign each Lagrangian point to a chordwise slice.
    //
    // Slicing rule:
    //   n = floor( (x - x_LE) / L * N_s )
    //
    // where:
    //   - L    : nondimensional chord length (typically 1.0)
    //   - N_s  : number of slices
    //
    // Each slice represents a control volume used later in
    // post-processing to integrate sectional forces.
    // ------------------------------------------------------------------
    d_slice_membership.resize(d_num_lag_pts, -1);

    for (int i = 0; i < d_num_lag_pts; ++i)
    {
        // Position relative to leading edge
        double x_rel = d_X_ref_x[i] - d_x_leading_edge;

        // Compute slice index
        int n = static_cast<int>(std::floor(x_rel * d_N_s / d_L));

        // Clamp to valid slice range
        if (n < 0)       n = 0;
        if (n >= d_N_s)  n = d_N_s - 1;

        d_slice_membership[i] = n;
    }

    return;
}


// ---------------------------------------------------------------------
// Static callback function registered with the IBAMR time integrator.
//
// This function is invoked once per timestep during integrateHierarchy().
// IBAMR calls this AFTER the constraint solve has been completed,
// meaning the Lagrange multiplier (constraint force) is now available.
//
// Purpose:
//   Bridge IBAMR's C-style callback interface to the
//   AddedMassDampingDiagnostics object instance.
//
// IMPORTANT:
//   - This function does NOT compute forces itself.
//   - It simply forwards control to the object method
//     captureConstraintForce(), passing the timestep size.
// ---------------------------------------------------------------------
void
AddedMassDampingDiagnostics::integrateHierarchyCallbackFcn(
    // Simulation time at the beginning of the timestep
    double current_time,

    // Simulation time at the end of the timestep
    double new_time,

    // Cycle number (unused here, required by IBAMR callback signature)
    int /*cycle_num*/,

    // Generic context pointer supplied at registration time.
    // This must point to an AddedMassDampingDiagnostics instance.
    void* ctx)
{
    // Cast the generic context pointer back to the diagnostics object.
    // This is safe because the pointer was registered by this class.
    auto* diag = static_cast<AddedMassDampingDiagnostics*>(ctx);

    // Capture the constraint force acting on the structure for this timestep.
    //
    // The timestep size dt = new_time - current_time is required to
    // convert the velocity correction (Lagrange multiplier) into force.
    //
    // This call:
    //   - Extracts the per-point constraint force from ConstraintIBMethod
    //   - Accumulates it into per-slice forces
    //   - Performs MPI reduction
    diag->captureConstraintForce(new_time - current_time);
}

void
AddedMassDampingDiagnostics::captureConstraintForce(double dt)
{
    // --------------------------------------------------------------
    // This routine collects the hydrodynamic constraint force acting
    // on the body from the ConstraintIBMethod at the current timestep
    // and bins it into chordwise slices.
    //
    // IMPORTANT DESIGN POINT:
    // - This function extracts ONLY solver-generated quantities.
    // - No analytical kinematics (omega, sin/cos, A(x), etc.) appear here.
    // - All forces are computed directly from the Lagrange multiplier.
    // --------------------------------------------------------------

    // Retrieve the Lagrange multiplier data from ConstraintIBMethod.
    // This represents the velocity correction used to enforce the
    // prescribed body motion.
    const std::vector<Pointer<IBTK::LData> >& lambda_data =
        d_ib_method_ops->getLagrangeMultiplier();

    // Access the Lagrangian data manager, which provides mappings
    // between Lagrangian indices, meshes, and hierarchy levels.
    IBTK::LDataManager* l_data_manager =
        d_ib_method_ops->getLDataManager();

    // Finest AMR level currently active in the hierarchy.
    const int finest_ln = d_patch_hierarchy->getFinestLevelNumber();

    // Local accumulator for per-slice force in the configured direction.
    // This will later be summed across MPI ranks.
    std::vector<double> F_L_n_local(d_N_s, 0.0);

    // --------------------------------------------------------------
    // Loop over all AMR levels that may contain Lagrangian data.
    // --------------------------------------------------------------
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        // Skip levels without Lagrangian data.
        if (!l_data_manager->levelContainsLagrangianData(ln)) continue;

        // Skip if no Lagrange multiplier exists on this level.
        if (static_cast<int>(lambda_data.size()) <= ln) continue;
        if (lambda_data[ln].isNull()) continue;

        // Access the local array of velocity corrections.
        // Each entry already includes the Lagrangian volume element.
        const boost::multi_array_ref<double, 2>& U_correction_data =
            *lambda_data[ln]->getLocalFormVecArray();

        // Get the Lagrangian mesh and the nodes local to this MPI rank.
        const Pointer<IBTK::LMesh> mesh = l_data_manager->getLMesh(ln);
        const std::vector<IBTK::LNode*>& local_nodes =
            mesh->getLocalNodes();

        // Determine the Lagrangian index range corresponding to
        // the structure handled by this diagnostics object.
        std::pair<int, int> lag_idx_range =
            l_data_manager->getLagrangianStructureIndexRange(
                d_structure_id, ln);

        const int struct_start = lag_idx_range.first;
        const int struct_end   = lag_idx_range.second;

        // ----------------------------------------------------------
        // Loop over all local Lagrangian nodes on this level.
        // ----------------------------------------------------------
        for (const auto& node : local_nodes)
        {
            // Global Lagrangian index of the node.
            const int lag_idx = node->getLagrangianIndex();

            // Process only nodes belonging to this structure.
            if (lag_idx >= struct_start && lag_idx < struct_end)
            {
                // Local PETSc index used to access U_correction_data.
                const int local_idx = node->getLocalPETScIndex();

                // Structure-local index (used for slice lookup).
                const int struct_local_idx = lag_idx - struct_start;

                // Determine which chordwise slice this point belongs to.
                int n = d_slice_membership[struct_local_idx];

                if (n >= 0 && n < d_N_s)
                {
                    // --------------------------------------------------
                    // Convert velocity correction into force:
                    //
                    // U_correction already includes the Lagrangian
                    // volume element. Dividing by dt and multiplying
                    // by rho gives the force on the FLUID.
                    //
                    // The force on the BODY is the negative of this.
                    //
                    // d_force_direction selects the component:
                    //   0 = x (e.g. cylinder inline oscillation)
                    //   1 = y (e.g. eel2d transverse oscillation)
                    // --------------------------------------------------
                    F_L_n_local[n] +=
                        -U_correction_data[local_idx][d_force_direction] * d_rho / dt;
                }
            }
        }

        // Restore the PETSc array for this level.
        lambda_data[ln]->restoreArrays();
    }

    // --------------------------------------------------------------
    // Perform a single MPI reduction to sum per-slice forces
    // across all MPI ranks.
    // --------------------------------------------------------------
    IBTK_MPI::sumReduction(&F_L_n_local[0], d_N_s);

    // Cache the reduced forces for output in computeAndWriteDiagnostics().
    for (int n = 0; n < d_N_s; ++n)
    {
        d_cached_F_L_n[n] = F_L_n_local[n];
    }

    // Mark that forces for this timestep have been successfully captured.
    d_force_captured = true;

    return;
}



} // namespace IBAMR
