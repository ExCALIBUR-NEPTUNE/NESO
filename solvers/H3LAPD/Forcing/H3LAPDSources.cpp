///////////////////////////////////////////////////////////////////////////////
//
// File: H3LAPDSources.cpp
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
// Description: Source/Forcing terms for the H3LAPD solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/core/ignore_unused.hpp>

#include "H3LAPDSources.h"

namespace Nektar {
std::string H3LAPDSources::className =
    SolverUtils::GetForcingFactory().RegisterCreatorFunction(
        "H3LAPDSources", H3LAPDSources::create,
        "Source terms for the H3LAPD solver");

H3LAPDSources::H3LAPDSources(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const std::weak_ptr<SolverUtils::EquationSystem> &pEquation)
    : Forcing(pSession, pEquation), field_to_index(pSession->GetVariables()) {}

void H3LAPDSources::v_InitObject(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
    const unsigned int &pNumForcingFields, const TiXmlElement *pForce) {
  boost::ignore_unused(pFields, pNumForcingFields, pForce);

  // Load parameters into class member vars
  // m_session->LoadParameter("example_param", m_egvar, 1.0);
}

void H3LAPDSources::v_Apply(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble &time) {
  boost::ignore_unused(pFields, inarray, outarray, time);
}

} // namespace Nektar
