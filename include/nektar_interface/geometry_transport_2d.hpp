/*
 *  Header only interface between Nektar++ and a particle code.
 */

#ifndef _CMMI_DEFS

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;
#define _MACRO_STRING(x) #x
#define STR(x) _MACRO_STRING(x)
#define MPICHK(cmd)                                                            \
  ASSERTL0(cmd == MPI_SUCCESS,                                                 \
           "MPI ERROR:" #cmd ":" STR(__LINE__) ":" __FILE__);

namespace NESO {

/*
 * Wrapper around a pair of inter and intra MPI Comms for shared memory
 * purposes.
 */
class CommPair {

private:
  bool allocated = false;

public:
  MPI_Comm comm_parent, comm_inter, comm_intra;
  int rank_parent, rank_inter, rank_intra;
  int size_parent, size_inter, size_intra;

  CommPair(){};

  CommPair(MPI_Comm comm_parent) {
    this->comm_parent = comm_parent;

    int rank_parent;
    MPICHK(MPI_Comm_rank(comm_parent, &rank_parent))
    MPICHK(MPI_Comm_split_type(comm_parent, MPI_COMM_TYPE_SHARED, 0,
                               MPI_INFO_NULL, &this->comm_intra))

    int rank_intra;
    MPICHK(MPI_Comm_rank(this->comm_intra, &rank_intra))
    const int colour_intra = (rank_intra == 0) ? 1 : MPI_UNDEFINED;
    MPICHK(MPI_Comm_split(comm_parent, colour_intra, 0, &this->comm_inter))

    this->allocated = true;

    MPICHK(MPI_Comm_rank(this->comm_parent, &this->rank_parent))
    MPICHK(MPI_Comm_rank(this->comm_intra, &this->rank_intra))
    MPICHK(MPI_Comm_size(this->comm_parent, &this->size_parent))
    MPICHK(MPI_Comm_size(this->comm_intra, &this->size_intra))
    if (comm_inter != MPI_COMM_NULL) {
      MPICHK(MPI_Comm_rank(this->comm_inter, &this->rank_inter))
      MPICHK(MPI_Comm_size(this->comm_inter, &this->size_inter))
    }
  };

  void Free() {
    int flag;
    MPICHK(MPI_Initialized(&flag))
    if (allocated && flag) {

      if ((this->comm_intra != MPI_COMM_NULL) &&
          (this->comm_intra != MPI_COMM_WORLD)) {
        MPICHK(MPI_Comm_free(&this->comm_intra))
        this->comm_intra = MPI_COMM_NULL;
      }

      if ((this->comm_inter != MPI_COMM_NULL) &&
          (this->comm_inter != MPI_COMM_WORLD)) {
        MPICHK(MPI_Comm_free(&this->comm_inter))
        this->comm_intra = MPI_COMM_NULL;
      }
    }
    this->allocated = false;
  }

  ~CommPair(){};
};

/*
 * Map from Linear indexed cartesian cell (lexicographic) to owning rank.
 */
class CartRankMap {

private:
public:
  CommPair comm_pair;
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int num_global_elements;
  MPI_Win map_win;
  int *map = NULL;
  int *map_base = NULL;

  int graph_dim;
  double local_bound_max[3];
  double local_bound_min[3];
  double global_bound_max[3];
  double global_bound_min[3];

  double cell_widths[3];
  int cell_counts[3];

  CartRankMap(){};
  ~CartRankMap(){};

  CartRankMap(LibUtilities::SessionReaderSharedPtr session,
              SpatialDomains::MeshGraphSharedPtr graph, CommPair comm_pair) {
    this->comm_pair = comm_pair;
    this->graph = graph;
    this->session = session;

    const int num_local_elements = graph->GetNumElements();

    MPICHK(MPI_Allreduce(&num_local_elements, &this->num_global_elements, 1,
                         MPI_INT, MPI_SUM, comm_pair.comm_parent))

    const MPI_Aint num_alloc = (comm_pair.rank_intra == 0)
                                   ? this->num_global_elements * sizeof(int)
                                   : 0;

    MPICHK(MPI_Win_allocate_shared(num_alloc, sizeof(int), MPI_INFO_NULL,
                                   comm_pair.comm_intra,
                                   (void *)&this->map_base, &this->map_win))

    MPI_Aint win_size_tmp;
    int disp_unit_tmp;

    MPICHK(MPI_Win_shared_query(this->map_win, 0, &win_size_tmp, &disp_unit_tmp,
                                (void *)&this->map))

    ASSERTL0(this->num_global_elements * sizeof(int) == win_size_tmp,
             "Pointer to incorrect size.")

    this->graph_dim = graph->GetMeshDimension();
    this->ComputeBoundingBox();
    this->ComputeCellMap();
  };

  void Free() {
    if (this->map_base != NULL) {
      MPICHK(MPI_Win_free(&this->map_win))
      this->map_base = NULL;
    }
  }

  void ComputeBoundingBox();
  int MapPointToCell(const double *point);
  void ComputeCellMap();
};

/*
 *  Map a point in space to a global linear cell index.
 */
int CartRankMap::MapPointToCell(const double *point) {
  int cells[3];
  for (int dimx = 0; dimx < this->graph_dim; dimx++) {
    const double shifted_origin = point[dimx] - this->global_bound_min[dimx];
    const double inverse_cell_width_dimx = 1.0 / this->cell_widths[dimx];
    const double double_bin = shifted_origin * inverse_cell_width_dimx;
    const int cell_dimx = (int)double_bin;
    cells[dimx] = cell_dimx;
  }

  int cell = cells[this->graph_dim - 1];
  for (int dimx = this->graph_dim - 2; dimx > -1; dimx--) {
    cell *= this->cell_counts[dimx];
    cell += cells[dimx];
  }

  return cell;
}

/*
 * Construct the global map from cell to owning rank in each shared memory
 * region.
 */
void CartRankMap::ComputeCellMap() {

  auto QuadGeoms = graph->GetAllQuadGeoms();

  // initialise the memory for the map
  if (comm_pair.rank_intra == 0) {
    for (int cellx = 0; cellx < this->num_global_elements; cellx++) {
      this->map[cellx] = -1;
    }
  }

  // record the cells on this rank as being owned by this rank in the
  // shared memory window
  MPICHK(MPI_Win_fence(0, this->map_win))
  MPICHK(MPI_Barrier(this->comm_pair.comm_intra))
  const int rank = comm_pair.rank_parent;

  Array<OneD, NekDouble> coords;
  for (auto const &quaditem : QuadGeoms) {
    const auto quad = quaditem.second;
    const auto num_verts = quad->GetNumVerts();
    const double inverse_num_verts = 1.0 / ((double)num_verts);

    double tmp_coords[3];

    for (int dimx = 0; dimx < graph_dim; dimx++) {
      tmp_coords[dimx] = 0.0;
    }

    for (int vx = 0; vx < num_verts; vx++) {
      const auto vert = quad->GetVertex(vx);
      vert->GetCoords(coords);
      for (int dimx = 0; dimx < graph_dim; dimx++) {
        tmp_coords[dimx] += inverse_num_verts * coords[dimx];
      }
    }

    const int cell = this->MapPointToCell(tmp_coords);
    // TODO record map from lexicographic cell to nektar++ cell
    this->map[cell] = rank;
  }

  MPICHK(MPI_Barrier(this->comm_pair.comm_intra))
  MPICHK(MPI_Win_fence(0, this->map_win))

  // reduce accross all shared regions (inter node)
  MPICHK(MPI_Win_fence(0, this->map_win))
  MPICHK(MPI_Barrier(this->comm_pair.comm_intra))

  if (comm_pair.rank_intra == 0) {
    int *tmp_map = (int *)malloc(this->num_global_elements * sizeof(int));
    ASSERTL0(tmp_map != NULL, "malloc failed");

    for (int cellx = 0; cellx < this->num_global_elements; cellx++) {
      tmp_map[cellx] = this->map[cellx];
    }

    MPICHK(MPI_Allreduce(tmp_map, this->map, this->num_global_elements, MPI_INT,
                         MPI_MAX, comm_pair.comm_inter))

    free(tmp_map);
  }

  MPICHK(MPI_Barrier(this->comm_pair.comm_intra))
  MPICHK(MPI_Win_fence(0, this->map_win))

  cout << "rank: " << comm_pair.rank_parent << endl;
  for (int cellx = 0; cellx < this->num_global_elements; cellx++) {
    cout << "cell: " << cellx << " owner: " << this->map[cellx] << endl;
  }

  // assert that all elements have an owning MPI rank
  if (comm_pair.rank_intra == 0) {
    for (int cellx = 0; cellx < this->num_global_elements; cellx++) {
      ASSERTL0(this->map[cellx] > -1, "Cell not claimed by any MPI rank");
    }
  }
}

/*
 *  Loop over the elements held on this MPI rank and use the vertices to
 *  compute the bounding box.
 */
void CartRankMap::ComputeBoundingBox() {
  ASSERTL0(graph->GetAllQuadGeoms().size() == graph->GetNumElements(),
           "Only implemented for Quads");

  auto QuadGeoms = graph->GetAllQuadGeoms();

  Array<OneD, NekDouble> coords;

  for (int dimx = 0; dimx < graph_dim; dimx++) {
    this->local_bound_min[dimx] = numeric_limits<NekDouble>::max();
    this->local_bound_max[dimx] = numeric_limits<NekDouble>::min();
    this->cell_widths[dimx] = numeric_limits<NekDouble>::min();
  }

  // TODO do the last vertices exist in the PBC case?

  for (auto const &quaditem : QuadGeoms) {
    const auto quad = quaditem.second;
    const auto num_verts = quad->GetNumVerts();

    double tmp_coords[3];
    double tmp_widths[3];
    for (int vx = 0; vx < num_verts; vx++) {
      const auto vert = quad->GetVertex(vx);

      vert->GetCoords(coords);
      for (int dimx = 0; dimx < graph_dim; dimx++) {
        local_bound_min[dimx] = min(local_bound_min[dimx], coords[dimx]);
        local_bound_max[dimx] = max(local_bound_max[dimx], coords[dimx]);

        if (vx == 0) {
          tmp_coords[dimx] = coords[dimx];
          tmp_widths[dimx] = numeric_limits<NekDouble>::min();
        } else {
          tmp_widths[dimx] =
              max(tmp_widths[dimx], abs(tmp_coords[dimx] - coords[dimx]));
        }
      }
    }

    for (int dimx = 0; dimx < graph_dim; dimx++) {
      this->cell_widths[dimx] = max(tmp_widths[dimx], this->cell_widths[dimx]);
    }
  }

  MPICHK(MPI_Allreduce(local_bound_min, this->global_bound_min, graph_dim,
                       MPI_DOUBLE, MPI_MIN, this->comm_pair.comm_parent))
  MPICHK(MPI_Allreduce(local_bound_max, this->global_bound_max, graph_dim,
                       MPI_DOUBLE, MPI_MAX, this->comm_pair.comm_parent))

  for (int dimx = 0; dimx < graph_dim; dimx++) {
    const double dim_width =
        this->global_bound_max[dimx] - this->global_bound_min[dimx];
    const int num_cells = rint(dim_width / this->cell_widths[dimx]);
    this->cell_counts[dimx] = num_cells;
  }
}

/*
 *  Interface between Nektar++ and a particle code for a Structured
 *  Cartesian mesh embedded in a linear unstructured Quad mesh.
 */
class MeshInterface {

private:
public:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  int graph_dim;

  MPI_Comm comm;
  int comm_size;
  int comm_rank;
  CommPair comm_pair;

  CartRankMap cart_rank_map;

  MeshInterface(LibUtilities::SessionReaderSharedPtr session,
                SpatialDomains::MeshGraphSharedPtr graph, MPI_Comm comm) {
    this->graph = graph;
    this->session = session;
    this->comm = comm;
    this->comm_pair = CommPair(comm);
    this->cart_rank_map = CartRankMap(session, graph, comm_pair);

    int flag;
    MPICHK(MPI_Initialized(&flag))
    ASSERTL0(flag == 1, "Expected MPI to be initialised.")

    MPICHK(MPI_Comm_rank(comm, &comm_rank))
    MPICHK(MPI_Comm_size(comm, &comm_size))

    auto nektar_comm = session->GetComm();
    ASSERTL0(nektar_comm->GetSize() == comm_size, "Comm size missmatch");
  }
  ~MeshInterface() {}

  void Free() {
    this->cart_rank_map.Free();
    this->comm_pair.Free();
  }
};

/*
 * Mirrors the existing PointGeom for packing.
 */
struct PointStruct {
  int coordim;
  int vid;
  NekDouble x;
  NekDouble y;
  NekDouble z;
};

/*
 *  General struct to hold the description of the arguments for SegGeoms and
 *  Curves.
 */
struct GeomPackSpec {
  int a;
  int b;
  int n_points;
};

template <class T> class GeomExtern : public T {
private:
protected:
public:
  SpatialDomains::SegGeomSharedPtr GetSegGeom(int index) {
    return this->m_edges[index];
  };
  SpatialDomains::CurveSharedPtr GetCurve() { return this->m_curve; };
};

template <typename T> class RemoteGeom2D {
public:
  int rank = -1;
  int id = -1;
  std::shared_ptr<T> geom;
  RemoteGeom2D(int rank, int id, std::shared_ptr<T> geom)
      : rank(rank), id(id), geom(geom){};
};

class PackedGeom2D {
private:
  // Push data onto the buffer.
  template <typename T> void push(T *data) {
    const size_t size = sizeof(T);
    const int offset_new = offset + size;
    buf.resize(offset_new);
    std::memcpy(((unsigned char *)buf.data()) + offset, data, size);
    offset = offset_new;
  }

  // Pop data from the buffer.
  template <typename T> void pop(T *data) {
    const size_t size = sizeof(T);
    const int offset_new = offset + size;
    ASSERTL0((offset_new <= input_length) || (input_length == -1),
             "Unserialiation overflows buffer.");
    std::memcpy(data, buf_in + offset, size);
    offset = offset_new;
  }

  /*
   * 2D Geoms are created with an id, edges and an optional curve. Edges
   * are constructed with an id, coordim, vertices and an optional curve.
   * Pack by looping over the edges and serialising each one into a
   * buffer.
   *
   */
  template <typename T> void pack_general(T &geom) {

    this->offset = 0;
    this->buf.reserve(512);
    this->id = geom.GetGlobalID();
    this->num_edges = geom.GetNumEdges();

    push(&rank);
    push(&local_id);
    push(&id);
    push(&num_edges);

    GeomPackSpec gs;
    PointStruct ps;

    for (int edgex = 0; edgex < num_edges; edgex++) {

      auto seg_geom = geom.GetSegGeom(edgex);
      const int num_points = seg_geom->GetNumVerts();

      gs.a = seg_geom->GetGlobalID();
      gs.b = seg_geom->GetCoordim();
      gs.n_points = num_points;
      push(&gs);

      for (int pointx = 0; pointx < num_points; pointx++) {
        auto point = seg_geom->GetVertex(pointx);
        ps.coordim = point->GetCoordim();
        ps.vid = point->GetVid();
        point->GetCoords(ps.x, ps.y, ps.z);
        push(&ps);
      }

      // curve of the edge
      auto curve = seg_geom->GetCurve();
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      push(&gs);
    }

    // curve of geom
    auto curve = geom.GetCurve();
    ASSERTL0(curve == nullptr, "Not implemented for curved edges");
    // A curve with n_points = -1 will be a taken as non-existant.
    gs.a = 0;
    gs.b = 0;
    gs.n_points = -1;
    push(&gs);
  };

  // Unpack the data common to both Quads and Triangles.
  void unpack_general() {
    ASSERTL0(offset == 0, "offset != 0 - cannot unpack twice");
    ASSERTL0(buf_in != nullptr, "source buffer has null pointer");

    // pop the metadata
    pop(&rank);
    pop(&local_id);
    pop(&id);
    pop(&num_edges);

    ASSERTL0(rank >= 0, "unreasonable rank");
    ASSERTL0((num_edges == 4) || (num_edges == 3),
             "Bad number of edges expected 4 or 3");

    GeomPackSpec gs;
    PointStruct ps;

    edges.reserve(num_edges);
    vertices.reserve(num_edges * 2);

    // pop and construct the edges
    for (int edgex = 0; edgex < num_edges; edgex++) {
      pop(&gs);
      const int edge_id = gs.a;
      const int edge_coordim = gs.b;
      const int n_points = gs.n_points;

      ASSERTL0(n_points >= 2, "Expected at least two points for an edge");

      const int points_offset = vertices.size();
      for (int pointx = 0; pointx < n_points; pointx++) {
        pop(&ps);
        vertices.push_back(std::make_shared<SpatialDomains::PointGeom>(
            ps.coordim, ps.vid, ps.x, ps.y, ps.z));
      }

      // In future the edge might have a corresponding curve
      pop(&gs);
      ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");

      // actually construct the edge
      auto edge_tmp = std::make_shared<SpatialDomains::SegGeom>(
          edge_id, edge_coordim, vertices.data() + points_offset);
      // edge_tmp->Setup();
      edges.push_back(edge_tmp);
    }

    // In future the geom might have a curve
    pop(&gs);
    ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");
  }

  /*
   *  Pack a 2DGeom into the buffer.
   */
  template <typename T> void pack(std::shared_ptr<T> &geom) {
    auto extern_geom = std::static_pointer_cast<GeomExtern<T>>(geom);
    this->pack_general(*extern_geom);
  }

  int rank;
  int local_id;
  int id;
  int num_edges;

  unsigned char *buf_in = nullptr;
  int input_length = 0;
  int offset = 0;

  std::vector<SpatialDomains::SegGeomSharedPtr> edges;
  std::vector<SpatialDomains::PointGeomSharedPtr> vertices;

public:
  std::vector<unsigned char> buf;

  PackedGeom2D(std::vector<unsigned char> &buf)
      : buf_in(buf.data()), input_length(buf.size()){};
  PackedGeom2D(unsigned char *buf_in, const int input_length = -1)
      : buf_in(buf_in), input_length(input_length){};

  template <typename T>
  PackedGeom2D(int rank, int local_id, std::shared_ptr<T> &geom) {
    this->rank = rank;
    this->local_id = local_id;
    pack(geom);
  }

  /*
   *  This offset is to help pointer arithmetric into the buffer for the next
   *  Geom.
   */
  int get_offset() { return this->offset; };
  /*
   * The rank that owns this geometry object.
   */
  int get_rank() { return this->rank; };
  /*
   * The local id of this geometry object on the remote rank.
   */
  int get_local_id() { return this->local_id; };

  /*
   *  Unpack the data as a 2DGeom.
   */
  template <typename T> std::shared_ptr<RemoteGeom2D<T>> unpack() {
    unpack_general();
    std::shared_ptr<T> geom = std::make_shared<T>(this->id, this->edges.data());
    geom->GetGeomFactors();
    geom->Setup();
    auto remote_geom = std::make_shared<RemoteGeom2D<T>>(rank, local_id, geom);
    return remote_geom;
  }
};

/*
 * Class to pack and unpack a set of 2D Geometry objects.
 *
 */
class PackedGeoms2D {
private:
  unsigned char *buf_in;
  int input_length = 0;
  int offset = 0;

  int get_packed_count() {
    int packed_count;
    ASSERTL0(input_length >= sizeof(int), "Input buffer has no count.");
    std::memcpy(&packed_count, buf_in, sizeof(int));
    ASSERTL0(((packed_count >= 0) && (packed_count <= input_length)),
             "Packed count is either negative or unrealistic.");
    return packed_count;
  }

public:
  std::vector<unsigned char> buf;

  PackedGeoms2D(){};

  /*
   * Pack a set of geometry objects collected by calling GetAllQuadGeoms or
   * GetAllTriGeoms on a MeshGraph object.
   */
  template <typename T>
  PackedGeoms2D(int rank, std::map<int, std::shared_ptr<T>> &geom_map) {
    const int num_geoms = geom_map.size();
    buf.reserve(512 * num_geoms);

    buf.resize(sizeof(int));
    std::memcpy(buf.data(), &num_geoms, sizeof(int));

    for (auto &geom_item : geom_map) {
      auto geom = geom_item.second;
      PackedGeom2D pg(rank, geom_item.first, geom);
      buf.insert(buf.end(), pg.buf.begin(), pg.buf.end());
    }
  };

  /*
   * Initialise this object in unpacking mode with a byte buffer of
   * serialised objects.
   */
  PackedGeoms2D(unsigned char *buf_in, int input_length) {
    this->offset = 0;
    this->buf_in = buf_in;
    this->input_length = input_length;
  };

  /*
   * Unserialise the held byte buffer as 2D geometry type T.
   */
  template <typename T>
  void unpack(std::vector<std::shared_ptr<RemoteGeom2D<T>>> &geoms) {
    const int packed_count = this->get_packed_count();
    geoms.reserve(geoms.size() + packed_count);
    for (int cx = 0; cx < packed_count; cx++) {
      auto new_packed_geom =
          PackedGeom2D(buf_in + sizeof(int) + offset, input_length);
      auto new_geom = new_packed_geom.unpack<T>();
      offset += new_packed_geom.get_offset();
      geoms.push_back(new_geom);
    }
    ASSERTL0(offset <= input_length, "buffer overflow occured");
  }
};

static inline int pos_mod(int i, int n) { return (i % n + n) % n; }

/*
 *  Collect all the 2D Geometry objects of type T from all the other MPI ranks.
 */
template <typename T>
std::vector<std::shared_ptr<RemoteGeom2D<T>>>
get_all_remote_geoms_2d(MPI_Comm comm,
                        std::map<int, std::shared_ptr<T>> &geom_map) {
  int rank, size;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  PackedGeoms2D local_packed_geoms(rank, geom_map);

  int max_buf_size, this_buf_size;
  this_buf_size = local_packed_geoms.buf.size();

  MPICHK(
      MPI_Allreduce(&this_buf_size, &max_buf_size, 1, MPI_INT, MPI_MAX, comm));

  // simplify send/recvs by choosing a buffer size of the maximum over all
  // ranks
  local_packed_geoms.buf.resize(max_buf_size);

  std::vector<unsigned char> buf_send(max_buf_size);
  std::vector<unsigned char> buf_recv(max_buf_size);

  memcpy(buf_send.data(), local_packed_geoms.buf.data(), max_buf_size);

  unsigned char *ptr_send = buf_send.data();
  unsigned char *ptr_recv = buf_recv.data();
  MPI_Status status;

  std::vector<std::shared_ptr<RemoteGeom2D<T>>> remote_geoms{};

  // cyclic passing of remote geometry objects.
  for (int shiftx = 0; shiftx < (size - 1); shiftx++) {

    int rank_send = pos_mod(rank + 1, size);
    int rank_recv = pos_mod(rank - 1, size);

    MPICHK(MPI_Sendrecv(ptr_send, max_buf_size, MPI_BYTE, rank_send, rank,
                        ptr_recv, max_buf_size, MPI_BYTE, rank_recv, rank_recv,
                        comm, &status));

    PackedGeoms2D remote_packed_geoms(ptr_recv, max_buf_size);
    remote_packed_geoms.unpack(remote_geoms);

    unsigned char *ptr_tmp = ptr_send;
    ptr_send = ptr_recv;
    ptr_recv = ptr_tmp;
  }

  return remote_geoms;
}

} // namespace NESO

#define _CNMI_DEFS
#endif
