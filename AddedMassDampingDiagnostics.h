// ---------------------------------------------------------------------
// AddedMassDampingDiagnostics.h
//
// Purpose:
//   Declare a lightweight IBAMR diagnostics class that outputs the
//   *per-slice hydrodynamic constraint force* acting on a structure.
//
// Design philosophy:
//   - The solver is responsible ONLY for quantities that depend on the
//     numerical solution (constraint forces).
//   - No analytical kinematics, wave parameters, or derived coefficients
//     (added mass, added damping) are computed here.
//   - All theory-based quantities are evaluated later in post-processing.
//
// Output produced by the implementation (.cpp):
//   time  F_L_0  F_L_1  ...  F_L_{N_s-1}
//
// where:
//   F_L_n(t) = constraint force on slice n in the configured direction.
//
// This separation ensures:
//   - Clean solver architecture
//   - Restart-safe output
//   - Consistency with ConstraintIBMethod and other IBAMR diagnostics
// ---------------------------------------------------------------------

#ifndef included_IBAMR_AddedMassDampingDiagnostics
#define included_IBAMR_AddedMassDampingDiagnostics

/////////////////////////////// INCLUDES /////////////////////////////////////

// ConstraintIBMethod provides access to Lagrange multipliers
// (constraint forces enforcing prescribed kinematics).
#include <ibamr/ConstraintIBMethod.h>

// PatchHierarchy gives access to the AMR hierarchy and finest level index,
// needed to loop over all Lagrangian data levels.
#include <PatchHierarchy.h>

// IBAMR/TBOX database interface for reading parameters from input2d.
#include <tbox/Database.h>

// Base class for lightweight utility/diagnostic objects.
#include <tbox/DescribedClass.h>

// Smart pointer wrapper used throughout IBAMR.
#include <tbox/Pointer.h>

// Standard C++ headers used for file output and data storage.
#include <fstream>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace IBAMR
{

/*!
 * \brief AddedMassDampingDiagnostics extracts and outputs per-slice
 * hydrodynamic constraint forces acting on a deforming immersed structure.
 *
 * -------------------------------------------------------------------------
 * ROLE IN THE SOLVER
 * -------------------------------------------------------------------------
 * This class is a *diagnostic utility* only. It does NOT:
 *   - prescribe kinematics,
 *   - compute analytical velocities or accelerations,
 *   - compute added-mass or damping coefficients.
 *
 * Instead, it outputs solver-native quantities that are later used
 * in post-processing.
 *
 * -------------------------------------------------------------------------
 * ARCHITECTURAL DESIGN (VERY IMPORTANT)
 * -------------------------------------------------------------------------
 * Solver responsibility:
 *   - Solve non-dimensional Navier-Stokes equations
 *   - Enforce prescribed kinematics via ConstraintIBMethod
 *   - Compute the resulting constraint force (Lagrange multiplier)
 *
 * This class:
 *   - Captures the constraint force acting ON the structure
 *   - Integrates it over chordwise slices
 *   - Writes per-slice forces to disk
 *
 * Post-processing responsibility:
 *   - Apply analytical kinematics (sin, cos, omega, k, A(x))
 *   - Compute added mass and added damping coefficients
 *
 * -------------------------------------------------------------------------
 * FORCE DIRECTION
 * -------------------------------------------------------------------------
 * The force_direction parameter (0=x, 1=y) selects which velocity
 * component of the Lagrange multiplier is extracted:
 *   - Eel2d (transverse oscillation): force_direction = 1 (default)
 *   - Oscillating cylinder (x-oscillation): force_direction = 0
 *
 * -------------------------------------------------------------------------
 * OUTPUT
 * -------------------------------------------------------------------------
 * Each timestep produces a single line:
 *
 *   time  F_L_0  F_L_1  ...  F_L_{N_s-1}
 *
 * where:
 *   - F_L_n(t) is the constraint force in the configured direction
 *     acting on slice n of the structure.
 *
 * These forces are NON-DIMENSIONAL and solver-consistent.
 *
 * -------------------------------------------------------------------------
 * DATA FLOW
 * -------------------------------------------------------------------------
 * ConstraintIBMethod
 *        |
 * Lagrange multiplier (velocity correction)
 *        |
 * captureConstraintForce()  [during integrateHierarchy]
 *        |
 * d_cached_F_L_n
 *        |
 * computeAndWriteDiagnostics()
 *        |
 * File output
 *
 * -------------------------------------------------------------------------
 * GEOMETRY HANDLING
 * -------------------------------------------------------------------------
 * - Slice membership is defined ONCE at initialization
 * - Uses the reference geometry from the .vertex file
 * - Independent of runtime deformation
 *
 * This guarantees that force integration is performed in a
 * fixed material (Lagrangian) frame.
 */
class AddedMassDampingDiagnostics : public SAMRAI::tbox::DescribedClass
{
public:
    /*!
     * \brief Constructor.
     *
     * Initializes solver-only parameters, reads reference geometry,
     * determines slice membership, and opens the output file.
     *
     * No analytical or kinematic quantities are read or computed here.
     */
    AddedMassDampingDiagnostics(
        const std::string& object_name,
        SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
        SAMRAI::tbox::Pointer<ConstraintIBMethod> ib_method_ops,
        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM> > patch_hierarchy);

    /*!
     * \brief Virtual destructor.
     *
     * Closes the output file on rank 0.
     * No solver-owned data is destroyed here.
     */
    virtual ~AddedMassDampingDiagnostics();

    /*!
     * \brief Write per-slice constraint forces for the current timestep.
     *
     * This function must be called AFTER advanceHierarchy().
     *
     * Output format:
     *   time  F_L_0  F_L_1  ...  F_L_{N_s-1}
     *
     * The values written here are later used in post-processing
     * to compute added mass and damping.
     */
    void computeAndWriteDiagnostics(double current_time);

    /*!
     * \brief Static callback registered with
     *        ConstraintIBMethod::registerIntegrateHierarchyCallback().
     *
     * This callback is invoked during integrateHierarchy(), while the
     * Lagrange multiplier data is still valid.
     *
     * Its sole purpose is to trigger captureConstraintForce().
     */
    static void integrateHierarchyCallbackFcn(double current_time,
                                               double new_time,
                                               int cycle_num,
                                               void* ctx);

private:
    // Disable default construction
    AddedMassDampingDiagnostics();

    // Disable copy construction
    AddedMassDampingDiagnostics(const AddedMassDampingDiagnostics&);

    // Disable assignment
    AddedMassDampingDiagnostics& operator=(const AddedMassDampingDiagnostics&);

    /*!
     * \brief Initialize chordwise slice membership for each Lagrangian point.
     *
     * Uses the reference geometry from the .vertex file and assigns
     * each point to a slice based on its x-location.
     *
     * This operation is performed ONCE and is time-independent.
     */
    void initializeSliceMembership();

    /*!
     * \brief Capture the constraint force acting on the structure.
     *
     * Extracts the Lagrange multiplier from ConstraintIBMethod,
     * converts it to physical force, and accumulates it per slice.
     *
     * This function must be called while the Lagrange multiplier
     * data is still alive (inside integrateHierarchy).
     */
    void captureConstraintForce(double dt);

    // ------------------------------------------------------------------
    // BASIC OBJECT DATA
    // ------------------------------------------------------------------

    // Descriptive name for logging and debugging
    std::string d_object_name;

    // Access to ConstraintIBMethod (source of constraint forces)
    SAMRAI::tbox::Pointer<ConstraintIBMethod> d_ib_method_ops;

    // Patch hierarchy (used to identify finest level and index ranges)
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM> > d_patch_hierarchy;

    // ------------------------------------------------------------------
    // SOLVER-ONLY PARAMETERS (NON-DIMENSIONAL)
    // ------------------------------------------------------------------

    // Chord length (typically L = 1 in non-dimensional formulation)
    double d_L;

    // Fluid density (consistent with RHO in input file)
    double d_rho;

    // Number of chordwise slices (numerical resolution parameter)
    int d_N_s;

    // Structure ID handled by this diagnostics object
    int d_structure_id;

    // Force extraction direction: 0=x, 1=y (default 1 for eel2d)
    // Cylinder x-oscillation uses 0; eel2d transverse oscillation uses 1.
    int d_force_direction;

    // ------------------------------------------------------------------
    // REFERENCE GEOMETRY (TIME-INDEPENDENT)
    // ------------------------------------------------------------------

    // Vertex file defining reference geometry
    std::string d_vertex_filename;

    // Reference x-coordinates of structure-local Lagrangian points
    std::vector<double> d_X_ref_x;

    // Mapping: Lagrangian point -> slice index
    std::vector<int> d_slice_membership;

    // Leading-edge x-coordinate (used as slice origin)
    double d_x_leading_edge;

    // Number of Lagrangian points in this structure
    int d_num_lag_pts;

    // ------------------------------------------------------------------
    // FORCE STORAGE
    // ------------------------------------------------------------------

    // Cached per-slice constraint force in the configured direction
    std::vector<double> d_cached_F_L_n;

    // Indicates whether force has been captured for the current timestep
    bool d_force_captured;

    // ------------------------------------------------------------------
    // OUTPUT
    // ------------------------------------------------------------------

    // Output filename (rank 0 only)
    std::string d_output_filename;

    // Output file stream
    std::ofstream d_out_file;

    // Flag indicating successful file open
    bool d_file_opened;
};

} // namespace IBAMR

#endif // included_IBAMR_AddedMassDampingDiagnostics
