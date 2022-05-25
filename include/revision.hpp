/// Information about the version of NESO
///
/// The build system will update this file on every commit, which may
/// result in files that include it getting rebuilt. Therefore it
/// should be included in as few places as possible

#ifndef NESO_REVISION_H
#define NESO_REVISION_H

namespace NESO {
namespace version {
/// The git commit hash
#ifndef NESO_REVISION
constexpr auto revision = "55f897954c1fceb7da86e2fff42f578efd68495c";
constexpr auto git_state = "CLEAN";
#else
// Stringify value passed at compile time
#define BUILDFLAG1_(x) #x
#define BUILDFLAG(x) BUILDFLAG1_(x)
constexpr auto revision = BUILDFLAG(NESO_REVISION);
constexpr auto git_state = BUILDFLAG(NESO_GIT_STATE);
#undef BUILDFLAG1
#undef BUILDFLAG
#endif
} // namespace version
} // namespace NESO

#endif // NESO_REVISION_H
