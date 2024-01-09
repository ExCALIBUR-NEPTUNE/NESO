///////////////////////////////////////////////////////////////////////////////
//
// File VariableConverter.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Auxiliary functions to convert variables in
//              the compressible flow system
//
///////////////////////////////////////////////////////////////////////////////
#include <iomanip>
#include <iostream>

#include <CompressibleFlowSolver/Misc/VariableConverter.h>
#include <LibUtilities/BasicUtils/Smath.hpp>
#include <LocalRegions/Expansion2D.h>
#include <LocalRegions/Expansion3D.h>

using namespace std;

namespace Nektar
{
VariableConverter::VariableConverter(
    const LibUtilities::SessionReaderSharedPtr &pSession, const int spaceDim,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : m_session(pSession), m_spacedim(spaceDim)
{
    // Create equation of state object
    std::string eosType;
    m_session->LoadSolverInfo("EquationOfState", eosType, "IdealGas");
    m_eos = GetEquationOfStateFactory().CreateInstance(eosType, m_session);

    // Parameters for dynamic viscosity
    m_session->LoadParameter("pInf", m_pInf, 101325);
    m_session->LoadParameter("rhoInf", m_rhoInf, 1.225);
    m_session->LoadParameter("GasConstant", m_gasConstant, 287.058);
    m_session->LoadParameter("mu", m_mu, 1.78e-05);
    m_oneOverT_star = (m_rhoInf * m_gasConstant) / m_pInf;

    // Parameters for sensor
    m_session->LoadParameter("Skappa", m_Skappa, -1.0);
    m_session->LoadParameter("Kappa", m_Kappa, 0.25);

    m_hOverP = NullNekDouble1DArray;

    // Shock sensor
    m_session->LoadSolverInfo("ShockCaptureType", m_shockCaptureType, "Off");
    if (m_shockCaptureType == "Physical")
    {
        // Artificial viscosity scaling constant
        m_session->LoadParameter("mu0", m_mu0, 1.0);

        m_muAv      = NullNekDouble1DArray;
        m_muAvTrace = NullNekDouble1DArray;

        // Check for Modal/Dilatation sensor
        m_session->LoadSolverInfo("ShockSensorType", m_shockSensorType,
                                  "Dilatation");

        // Check for Ducros sensor
        m_session->LoadSolverInfo("DucrosSensor", m_ducrosSensor, "Off");

        if (m_ducrosSensor != "Off" || m_shockSensorType == "Dilatation")
        {
            m_flagCalcDivCurl = true;
        }
    }
    // Load smoothing tipe
    m_session->LoadSolverInfo("Smoothing", m_smoothing, "Off");
    if (m_smoothing == "C0")
    {
        m_C0ProjectExp =
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                m_session, pGraph, m_session->GetVariable(0));
    }

    std::string viscosityType;
    m_session->LoadSolverInfo("ViscosityType", viscosityType, "Constant");
    if ("Variable" == viscosityType)
    {
        WARNINGL0(
            m_session->DefinesParameter("Tref"),
            "The Tref should be given in Kelvin for using the Sutherland's law "
            "of dynamic viscosity. The default is 288.15. Note the mu or "
            "Reynolds number should coorespond to this temperature.");
        m_session->LoadParameter("Tref", m_Tref, 288.15);
        m_TRatioSutherland = 110.0 / m_Tref;
    }
}

/**
 * @brief Destructor for VariableConverter class.
 */
VariableConverter::~VariableConverter()
{
}

/**
 * @brief Compute the dynamic energy
 *        \f$ e = rho*V^2/2 \f$.
 */
void VariableConverter::GetDynamicEnergy(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &energy)
{
    size_t nPts = physfield[m_spacedim + 1].size();
    Vmath::Zero(nPts, energy, 1);

    // tmp = (rho * u_i)^2
    for (int i = 0; i < m_spacedim; ++i)
    {
        Vmath::Vvtvp(nPts, physfield[i + 1], 1, physfield[i + 1], 1, energy, 1,
                     energy, 1);
    }
    // Divide by rho and multiply by 0.5 --> tmp = 0.5 * rho * u^2
    Vmath::Vdiv(nPts, energy, 1, physfield[0], 1, energy, 1);
    Vmath::Smul(nPts, 0.5, energy, 1, energy, 1);
}

/**
 * @brief Compute the specific internal energy
 *        \f$ e = (E - rho*V^2/2)/rho \f$.
 */
void VariableConverter::GetInternalEnergy(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &energy)
{
    int nPts = physfield[0].size();
    Array<OneD, NekDouble> tmp(nPts);

    GetDynamicEnergy(physfield, tmp);

    // Calculate rhoe = E - rho*V^2/2
    Vmath::Vsub(nPts, physfield[m_spacedim + 1], 1, tmp, 1, energy, 1);
    // Divide by rho
    Vmath::Vdiv(nPts, energy, 1, physfield[0], 1, energy, 1);
}

/**
 * @brief Compute the specific enthalpy \f$ h = e + p/rho \f$.
 */
void VariableConverter::GetEnthalpy(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &enthalpy)
{
    int nPts = physfield[0].size();
    Array<OneD, NekDouble> energy(nPts, 0.0);
    Array<OneD, NekDouble> pressure(nPts, 0.0);

    GetInternalEnergy(physfield, energy);
    GetPressure(physfield, pressure);

    // Calculate p/rho
    Vmath::Vdiv(nPts, pressure, 1, physfield[0], 1, enthalpy, 1);
    // Calculate h = e + p/rho
    Vmath::Vadd(nPts, energy, 1, enthalpy, 1, enthalpy, 1);
}

/**
 * @brief Compute the velocity field \f$ \mathbf{v} \f$ given the momentum
 * \f$ \rho\mathbf{v} \f$.
 *
 * @param physfield  Momentum field.
 * @param velocity   Velocity field.
 */
void VariableConverter::GetVelocityVector(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, NekDouble>> &velocity)
{
    const int nPts = physfield[0].size();

    for (int i = 0; i < m_spacedim; ++i)
    {
        Vmath::Vdiv(nPts, physfield[1 + i], 1, physfield[0], 1, velocity[i], 1);
    }
}

/**
 * @brief Compute the mach number \f$ M = \| \mathbf{v} \|^2 / c \f$.
 *
 * @param physfield    Input physical field.
 * @param soundfield   The speed of sound corresponding to physfield.
 * @param mach         The resulting mach number \f$ M \f$.
 */
void VariableConverter::GetMach(Array<OneD, Array<OneD, NekDouble>> &physfield,
                                Array<OneD, NekDouble> &soundspeed,
                                Array<OneD, NekDouble> &mach)
{
    const int nPts = physfield[0].size();

    Vmath::Vmul(nPts, physfield[1], 1, physfield[1], 1, mach, 1);

    for (int i = 1; i < m_spacedim; ++i)
    {
        Vmath::Vvtvp(nPts, physfield[1 + i], 1, physfield[1 + i], 1, mach, 1,
                     mach, 1);
    }

    Vmath::Vdiv(nPts, mach, 1, physfield[0], 1, mach, 1);
    Vmath::Vdiv(nPts, mach, 1, physfield[0], 1, mach, 1);
    Vmath::Vsqrt(nPts, mach, 1, mach, 1);

    Vmath::Vdiv(nPts, mach, 1, soundspeed, 1, mach, 1);
}

/**
 * @brief Compute the dynamic viscosity using the Sutherland's law
 * \f$ \mu = \mu_star * (T / T_star)^3/2 * (1 + C) / (T/T_star + C) \f$,
 *  C      : 110. /Tref
 *  Tref   : the reference temperature, Tref, should always given in Kelvin,
 *           if non-dimensional should be the reference for non-dimensionalizing
 *  muref  : the dynamic viscosity or the 1/Re corresponding to Tref
 *  T_star : m_pInf / (m_rhoInf * m_gasConstant),non-dimensional or dimensional
 *
 * WARNING, if this routine is modified the same must be done in the
 * FieldConvert utility ProcessWSS.cpp (this class should be restructured).
 *
 * @param temperature  Input temperature.
 * @param mu           The resulting dynamic viscosity.
 */
void VariableConverter::GetDynamicViscosity(
    const Array<OneD, const NekDouble> &temperature, Array<OneD, NekDouble> &mu)
{
    const int nPts = temperature.size();

    for (int i = 0; i < nPts; ++i)
    {
        mu[i] = GetDynamicViscosity(temperature[i]);
    }
}

/**
 * @brief Compute the dynamic viscosity using the Sutherland's law
 * \f$ \mu = \mu_star * (T / T_star)^3/2 * (T_star + 110) / (T + 110) \f$,
 */
void VariableConverter::GetDmuDT(
    const Array<OneD, const NekDouble> &temperature,
    const Array<OneD, const NekDouble> &mu, Array<OneD, NekDouble> &DmuDT)
{
    const int nPts = temperature.size();
    NekDouble tmp  = 0.0;
    NekDouble ratio;

    for (int i = 0; i < nPts; ++i)
    {
        ratio = temperature[i] * m_oneOverT_star;
        tmp   = 0.5 * (ratio + 3.0 * m_TRatioSutherland) /
              (ratio * (ratio + m_TRatioSutherland));
        DmuDT[i] = mu[i] * tmp * m_oneOverT_star;
    }
}

void VariableConverter::GetAbsoluteVelocity(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &Vtot)
{
    const int nPts = physfield[0].size();

    // Getting the velocity vector on the 2D normal space
    Array<OneD, Array<OneD, NekDouble>> velocity(m_spacedim);

    Vmath::Zero(Vtot.size(), Vtot, 1);

    for (int i = 0; i < m_spacedim; ++i)
    {
        velocity[i] = Array<OneD, NekDouble>(nPts);
    }

    GetVelocityVector(physfield, velocity);

    for (int i = 0; i < m_spacedim; ++i)
    {
        Vmath::Vvtvp(nPts, velocity[i], 1, velocity[i], 1, Vtot, 1, Vtot, 1);
    }

    Vmath::Vsqrt(nPts, Vtot, 1, Vtot, 1);
}

/**
 * @brief Calculate the pressure using the equation of state.
 *
 * @param physfield  Input momentum.
 * @param pressure   Computed pressure field.
 */
void VariableConverter::GetPressure(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &pressure)
{
    int nPts = physfield[0].size();

    Array<OneD, NekDouble> energy(nPts);
    GetInternalEnergy(physfield, energy);

    for (int i = 0; i < nPts; ++i)
    {
        pressure[i] = m_eos->GetPressure(physfield[0][i], energy[i]);
    }
}

/**
 * @brief Compute the temperature using the equation of state.
 *
 * @param physfield    Input physical field.
 * @param temperature  The resulting temperature \f$ T \f$.
 */
void VariableConverter::GetTemperature(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &temperature)
{
    int nPts = physfield[0].size();

    Array<OneD, NekDouble> energy(nPts);
    GetInternalEnergy(physfield, energy);

    for (int i = 0; i < nPts; ++i)
    {
        temperature[i] = m_eos->GetTemperature(physfield[0][i], energy[i]);
    }
}

/**
 * @brief Compute the sound speed using the equation of state.
 *
 * @param physfield    Input physical field
 * @param soundspeed   The resulting sound speed \f$ c \f$.
 */
void VariableConverter::GetSoundSpeed(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &soundspeed)
{
    int nPts = physfield[0].size();

    Array<OneD, NekDouble> energy(nPts);
    GetInternalEnergy(physfield, energy);

    for (int i = 0; i < nPts; ++i)
    {
        soundspeed[i] = m_eos->GetSoundSpeed(physfield[0][i], energy[i]);
    }
}

/**
 * @brief Compute the entropy using the equation of state.
 *
 * @param physfield    Input physical field
 * @param soundspeed   The resulting sound speed \f$ c \f$.
 */
void VariableConverter::GetEntropy(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &entropy)
{
    int nPts = physfield[0].size();

    Array<OneD, NekDouble> energy(nPts);
    GetInternalEnergy(physfield, energy);

    for (int i = 0; i < nPts; ++i)
    {
        entropy[i] = m_eos->GetEntropy(physfield[0][i], energy[i]);
    }
}

/**
 * @brief Compute \f$ e(rho,p) \f$ using the equation of state.
 *
 * @param rho          Input density
 * @param pressure     Input pressure
 * @param energy       The resulting internal energy.
 */
void VariableConverter::GetEFromRhoP(const Array<OneD, NekDouble> &rho,
                                     const Array<OneD, NekDouble> &pressure,
                                     Array<OneD, NekDouble> &energy)
{
    int nPts = rho.size();

    for (int i = 0; i < nPts; ++i)
    {
        energy[i] = m_eos->GetEFromRhoP(rho[i], pressure[i]);
    }
}

/**
 * @brief Compute \f$ rho(p,T) \f$ using the equation of state.
 *
 * @param pressure     Input pressure
 * @param temperature  Input temperature
 * @param rho          The resulting density
 */
void VariableConverter::GetRhoFromPT(const Array<OneD, NekDouble> &pressure,
                                     const Array<OneD, NekDouble> &temperature,
                                     Array<OneD, NekDouble> &rho)
{
    int nPts = pressure.size();

    for (int i = 0; i < nPts; ++i)
    {
        rho[i] = m_eos->GetRhoFromPT(pressure[i], temperature[i]);
    }
}

void VariableConverter::SetAv(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, const Array<OneD, NekDouble>> &consVar,
    const Array<OneD, NekDouble> &div,
    const Array<OneD, NekDouble> &curlSquared)
{
    auto nTracePts = fields[0]->GetTrace()->GetTotPoints();
    if (m_muAv == NullNekDouble1DArray)
    {
        auto nPts   = fields[0]->GetTotPoints();
        m_muAv      = Array<OneD, NekDouble>(nPts, 0.0);
        m_muAvTrace = Array<OneD, NekDouble>(nTracePts, 0.0);
        SetElmtMinHP(fields);
    }

    if (m_shockSensorType == "Modal")
    {
        // Get viscosity based on modal sensor
        GetMuAv(fields, consVar, m_muAv);
    }
    else
    {
        // Get viscosity based on dilatation sensor
        GetMuAv(fields, consVar, div, m_muAv);
    }

    // Apply Ducros sensor
    if (m_ducrosSensor != "Off")
    {
        ApplyDucros(div, curlSquared, m_muAv);
    }

    // Apply approximate C0 smoothing
    if (m_smoothing == "C0")
    {
        ApplyC0Smooth(m_muAv);
    }

    // Set trace AV
    Array<OneD, NekDouble> muFwd(nTracePts, 0.0), muBwd(nTracePts, 0.0);
    fields[0]->GetFwdBwdTracePhys(m_muAv, muFwd, muBwd, false, false, false);
    for (size_t p = 0; p < nTracePts; ++p)
    {
        m_muAvTrace[p] = 0.5 * (muFwd[p] + muBwd[p]);
    }
}

Array<OneD, NekDouble> &VariableConverter::GetAv()
{
    ASSERTL1(m_muAv != NullNekDouble1DArray, "m_muAv not set");
    return m_muAv;
}

Array<OneD, NekDouble> &VariableConverter::GetAvTrace()
{
    ASSERTL1(m_muAvTrace != NullNekDouble1DArray, "m_muAvTrace not set");
    return m_muAvTrace;
}

/**
 * @brief Compute an estimate of minimum h/p
 * for each element of the expansion.
 */
void VariableConverter::SetElmtMinHP(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
    auto nElements = fields[0]->GetExpSize();
    if (m_hOverP == NullNekDouble1DArray)
    {
        m_hOverP = Array<OneD, NekDouble>(nElements, 1.0);
    }

    // Determine h/p scaling
    Array<OneD, int> pOrderElmt = fields[0]->EvalBasisNumModesMaxPerExp();
    auto expdim                 = fields[0]->GetGraph()->GetMeshDimension();
    for (size_t e = 0; e < nElements; e++)
    {
        NekDouble h = 1.0e+10;
        switch (expdim)
        {
            case 3:
            {
                LocalRegions::Expansion3DSharedPtr exp3D;
                exp3D = fields[0]->GetExp(e)->as<LocalRegions::Expansion3D>();
                for (size_t i = 0; i < exp3D->GetNtraces(); ++i)
                {
                    h = min(
                        h, exp3D->GetGeom3D()->GetEdge(i)->GetVertex(0)->dist(*(
                               exp3D->GetGeom3D()->GetEdge(i)->GetVertex(1))));
                }
                break;
            }

            case 2:
            {
                LocalRegions::Expansion2DSharedPtr exp2D;
                exp2D = fields[0]->GetExp(e)->as<LocalRegions::Expansion2D>();
                for (size_t i = 0; i < exp2D->GetNtraces(); ++i)
                {
                    h = min(
                        h, exp2D->GetGeom2D()->GetEdge(i)->GetVertex(0)->dist(*(
                               exp2D->GetGeom2D()->GetEdge(i)->GetVertex(1))));
                }
                break;
            }
            case 1:
            {
                LocalRegions::Expansion1DSharedPtr exp1D;
                exp1D = fields[0]->GetExp(e)->as<LocalRegions::Expansion1D>();

                h = min(h, exp1D->GetGeom1D()->GetVertex(0)->dist(
                               *(exp1D->GetGeom1D()->GetVertex(1))));

                break;
            }
            default:
            {
                ASSERTL0(false, "Dimension out of bound.")
            }
        }

        // Store h/p scaling
        m_hOverP[e] = h / max(pOrderElmt[e] - 1, 1);
    }
}

Array<OneD, NekDouble> &VariableConverter::GetElmtMinHP()
{
    ASSERTL1(m_hOverP != NullNekDouble1DArray, "m_hOverP not set");
    return m_hOverP;
}

void VariableConverter::GetSensor(
    const MultiRegions::ExpListSharedPtr &field,
    const Array<OneD, const Array<OneD, NekDouble>> &physarray,
    Array<OneD, NekDouble> &Sensor, Array<OneD, NekDouble> &SensorKappa,
    int offset)
{
    NekDouble Skappa;
    NekDouble order;
    Array<OneD, NekDouble> tmp;
    Array<OneD, int> expOrderElement = field->EvalBasisNumModesMaxPerExp();

    for (int e = 0; e < field->GetExpSize(); e++)
    {
        int numModesElement = expOrderElement[e];
        int nElmtPoints     = field->GetExp(e)->GetTotPoints();
        int physOffset      = field->GetPhys_Offset(e);
        int nElmtCoeffs     = field->GetExp(e)->GetNcoeffs();
        int numCutOff       = numModesElement - offset;

        if (numModesElement <= offset)
        {
            Vmath::Fill(nElmtPoints, 0.0, tmp = Sensor + physOffset, 1);
            Vmath::Fill(nElmtPoints, 0.0, tmp = SensorKappa + physOffset, 1);
            continue;
        }

        // create vector to save the solution points per element at P = p;
        Array<OneD, NekDouble> elmtPhys(nElmtPoints,
                                        tmp = physarray[0] + physOffset);
        // Compute coefficients
        Array<OneD, NekDouble> elmtCoeffs(nElmtCoeffs, 0.0);
        field->GetExp(e)->FwdTrans(elmtPhys, elmtCoeffs);

        // ReduceOrderCoeffs reduces the polynomial order of the solution
        // that is represented by the coeffs given as an inarray. This is
        // done by projecting the higher order solution onto the orthogonal
        // basis and padding the higher order coefficients with zeros.
        Array<OneD, NekDouble> reducedElmtCoeffs(nElmtCoeffs, 0.0);
        field->GetExp(e)->ReduceOrderCoeffs(numCutOff, elmtCoeffs,
                                            reducedElmtCoeffs);

        Array<OneD, NekDouble> reducedElmtPhys(nElmtPoints, 0.0);
        field->GetExp(e)->BwdTrans(reducedElmtCoeffs, reducedElmtPhys);

        NekDouble numerator   = 0.0;
        NekDouble denominator = 0.0;

        // Determining the norm of the numerator of the Sensor
        Array<OneD, NekDouble> difference(nElmtPoints, 0.0);
        Vmath::Vsub(nElmtPoints, elmtPhys, 1, reducedElmtPhys, 1, difference,
                    1);

        numerator   = Vmath::Dot(nElmtPoints, difference, difference);
        denominator = Vmath::Dot(nElmtPoints, elmtPhys, elmtPhys);

        NekDouble elmtSensor = sqrt(numerator / denominator);
        elmtSensor = log10(max(elmtSensor, NekConstants::kNekSqrtTol));

        Vmath::Fill(nElmtPoints, elmtSensor, tmp = Sensor + physOffset, 1);

        // Compute reference value for sensor
        order = max(numModesElement - 1, 1);
        if (order > 0)
        {
            Skappa = m_Skappa - 4.25 * log10(static_cast<NekDouble>(order));
        }
        else
        {
            Skappa = 0.0;
        }

        // Compute artificial viscosity
        NekDouble elmtSensorKappa;
        if (elmtSensor < (Skappa - m_Kappa))
        {
            elmtSensorKappa = 0;
        }
        else if (elmtSensor > (Skappa + m_Kappa))
        {
            elmtSensorKappa = 1.0;
        }
        else
        {
            elmtSensorKappa =
                0.5 * (1 + sin(M_PI * (elmtSensor - Skappa) / (2 * m_Kappa)));
        }
        Vmath::Fill(nElmtPoints, elmtSensorKappa,
                    tmp = SensorKappa + physOffset, 1);
    }
}

/**
 * @brief Calculate the physical artificial viscosity based on modal sensor.
 *
 * @param consVar  Input field.
 */
void VariableConverter::GetMuAv(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, const Array<OneD, NekDouble>> &consVar,
    Array<OneD, NekDouble> &muAv)
{
    auto nPts = consVar[0].size();
    // Determine the maximum wavespeed
    Array<OneD, NekDouble> Lambdas(nPts, 0.0);
    Array<OneD, NekDouble> soundspeed(nPts, 0.0);
    Array<OneD, NekDouble> absVelocity(nPts, 0.0);
    GetSoundSpeed(consVar, soundspeed);
    GetAbsoluteVelocity(consVar, absVelocity);
    Vmath::Vadd(nPts, absVelocity, 1, soundspeed, 1, Lambdas, 1);

    // Compute sensor based on rho
    Array<OneD, NekDouble> Sensor(nPts, 0.0);
    GetSensor(fields[0], consVar, Sensor, muAv, 1);

    Array<OneD, NekDouble> tmp;
    auto nElmt = fields[0]->GetExpSize();
    for (size_t e = 0; e < nElmt; ++e)
    {
        auto physOffset  = fields[0]->GetPhys_Offset(e);
        auto nElmtPoints = fields[0]->GetExp(e)->GetTotPoints();

        // Compute the maximum wave speed
        NekDouble LambdaElmt = 0.0;
        LambdaElmt = Vmath::Vmax(nElmtPoints, tmp = Lambdas + physOffset, 1);

        // Compute average bounded density
        NekDouble rhoAve =
            Vmath::Vsum(nElmtPoints, tmp = consVar[0] + physOffset, 1);
        rhoAve = rhoAve / nElmtPoints;
        rhoAve = Smath::Smax(rhoAve, 1.0e-4, 1.0e+4);

        // Scale sensor by coeff, h/p, and density
        LambdaElmt *= m_mu0 * m_hOverP[e] * rhoAve;
        Vmath::Smul(nElmtPoints, LambdaElmt, tmp = muAv + physOffset, 1,
                    tmp = muAv + physOffset, 1);
    }
}

/**
 * @brief Calculate the physical artificial viscosity based on dilatation of
 * velocity vector.
 *
 * @param
 */
void VariableConverter::GetMuAv(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, const Array<OneD, NekDouble>> &consVar,
    const Array<OneD, NekDouble> &div, Array<OneD, NekDouble> &muAv)
{
    auto nPts = consVar[0].size();

    // Get sound speed
    // theoretically it should be used  the critical sound speed, this
    // matters for large Mach numbers (above 3.0)
    Array<OneD, NekDouble> soundSpeed(nPts, 0.0);
    GetSoundSpeed(consVar, soundSpeed);

    // Get abosolute velocity to compute lambda
    Array<OneD, NekDouble> absVelocity(nPts, 0.0);
    GetAbsoluteVelocity(consVar, absVelocity);

    // Loop over elements
    auto nElmt = fields[0]->GetExpSize();
    for (size_t e = 0; e < nElmt; ++e)
    {
        auto nElmtPoints = fields[0]->GetExp(e)->GetTotPoints();
        auto physOffset  = fields[0]->GetPhys_Offset(e);
        auto physEnd     = physOffset + nElmtPoints;

        NekDouble hOpTmp = m_hOverP[e];

        // Loop over the points
        for (size_t p = physOffset; p < physEnd; ++p)
        {
            // Get non-dimensional sensor based on dilatation
            NekDouble sSpeedTmp = soundSpeed[p];
            // (only compression waves)
            NekDouble divTmp = -div[p];
            divTmp           = std::max(divTmp, 0.0);
            NekDouble sensor = m_mu0 * hOpTmp * divTmp / sSpeedTmp;
            // Scale to viscosity scale
            NekDouble rho    = consVar[0][p];
            NekDouble lambda = sSpeedTmp + absVelocity[p];
            muAv[p]          = sensor * rho * lambda * hOpTmp;
        }
    }
}

/**
 * @brief Apply Ducros (anti-vorticity) sensor averaged over the element.
 *
 * @param field Input Field
 */
void VariableConverter::ApplyDucros(const Array<OneD, NekDouble> &div,
                                    const Array<OneD, NekDouble> &curlSquare,
                                    Array<OneD, NekDouble> &muAv)
{
    // machine eps**2
    NekDouble eps = std::numeric_limits<NekDouble>::epsilon();
    eps *= eps;

    // loop over points
    auto nPts = div.size();
    for (size_t p = 0; p < nPts; ++p)
    {
        NekDouble tmpDiv2 = div[p];
        tmpDiv2 *= tmpDiv2;
        NekDouble denDuc = tmpDiv2 + curlSquare[p] + eps;
        NekDouble Duc    = tmpDiv2 / denDuc;
        // apply
        muAv[p] *= Duc;
    }
}

/**
 * @brief Make field C0.
 *
 * @param field Input Field
 */
void VariableConverter::ApplyC0Smooth(Array<OneD, NekDouble> &field)
{
    auto nCoeffs = m_C0ProjectExp->GetNcoeffs();
    Array<OneD, NekDouble> muFwd(nCoeffs);
    Array<OneD, NekDouble> weights(nCoeffs, 1.0);
    // Assemble global expansion coefficients for viscosity
    m_C0ProjectExp->FwdTransLocalElmt(field, m_C0ProjectExp->UpdateCoeffs());
    m_C0ProjectExp->Assemble();
    Vmath::Vcopy(nCoeffs, m_C0ProjectExp->GetCoeffs(), 1, muFwd, 1);
    // Global coefficients
    Vmath::Vcopy(nCoeffs, weights, 1, m_C0ProjectExp->UpdateCoeffs(), 1);
    // This is the sign vector
    m_C0ProjectExp->GlobalToLocal();
    // Get weights
    m_C0ProjectExp->Assemble();
    // Divide
    Vmath::Vdiv(nCoeffs, muFwd, 1, m_C0ProjectExp->GetCoeffs(), 1,
                m_C0ProjectExp->UpdateCoeffs(), 1);
    // Get local coefficients
    m_C0ProjectExp->GlobalToLocal();
    // Get C0 field
    m_C0ProjectExp->BwdTrans(m_C0ProjectExp->GetCoeffs(), field);
}

} // namespace Nektar
