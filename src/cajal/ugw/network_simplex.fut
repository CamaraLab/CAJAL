import "tree2"

module type network_simplex_context = {
  module tree : tree
  include WeightedDiGraph with node = tree.index
  val num_bits : i32
  val get_bit : i32 -> N.t -> i32
  type direction = #up | #down
  val parent_arc : node -> (arc , direction)
  type node_potentials[k] = tree.data [k] N.t
}

-- module network_simplex_impl (N : numeric) : network_simplex_context = {
--   open tree_impl
--   type node = tree_index
--   def cost (i: tree_index) (j: tree_index) = A[i][j]
--   -- The node indices are assumed to be labelled by the integers 0...n-1.
--   -- The edges are stored as a list of triples (i, j, k).
--   type graph[k] = [k](i64,  i64, i64)
--   type arc = (i64, i64)
--   type node_data = { pred : i64, depth : i64, succ : i64 }
--   -- The first element is the root, the second one is
--   -- an array where the k-th node (in some fixed ordering?)
--   -- is associated to the k-th index.
--   type spanning_tree [k] = (i64, ([k]node_data) )
--   type node_potentials [k] = tree_data [k] i64
-- }

module network_simplex(M : network_simplex_context) = {
  -- N : Numeric
  module N = M.N
  type index = M.tree.index
  type t = N.t

  import "lib/github.com/diku-dk/sorts/radix_sort"
  import "lib/github.com/diku-dk/segmented/segmented"

  def xor (x: bool) (y: bool) = bool.(x != y)
  def begin_streak [n] (b: [n]bool) : [n]i64 =
    let diffs =
      ([false] ++ map2 xor (map (\i -> b[i]) (0..<(n-1))) (map (\i -> b[i+1])
							       (0..<(n-1)))) :> [n]bool
    in
    let a = map (\i -> if diffs[i] then i else 0) (iota n) in
    segmented_scan (i64.+) (0) diffs a

  -- def initial_spanning_tree_ot [n][m] (mu: [n]N.t) (nu: [m]N.t) :
  --  (M.tree.structure[(n+m)], M.tree.data[(n+m)]N.t) =
  --   let mu_cum = scan (N.+) (N.i32 0) mu in
  --   let nu_cum = scan (N.+) (N.i32 0) nu in
  --   let f (i:i32 , t) =
  --     if i == 0 then (N.i32 0) else
  --     if t then nu_cum[i-1] else mu_cum[i-1]
  --   in
  --   let Z1 = zip (replicate n false) (0..<n) in
  --   let Z2 = zip (replicate m true) (0..<m) in
  --   let X = radix_sort_by_key f (M.num_bits) (M.get_bit) (Z1 ++ Z2) in
  --   let (indices,partition) = unzip X in


  -- This function constructs an initial spanning tree on
  -- the graph arising from an optimal transport problem.
  -- In optimal transport, we have sets A and B,
  -- the edges are exactly given by pairs (i,j) with i in A and j in B,
  -- and flows are unbounded.
  -- The underlying
  def initial_spanning_tree_ot [n][m] (mu: [n]N.t) (nu: [m]N.t) :
   M.tree.structure[(n+m)] =
    let f (i: i64) =
      let mu_cum = scan (N.+) (N.i32 0) mu in
      let nu_cum = scan (N.+) (N.i32 0) nu in
      if i == 0 then (N.i32 0) else
      if i == n then (N.i32 0) else
      if i < n then mu_cum[i-1] else
	nu_cum[(i-n)-1]
    in
    let Y = 
      let X = radix_sort_by_key f (M.num_bits) (M.get_bit) (0..<(n+m)) in
      begin_streak (map (\i -> i < n) X) |> map (\i -> i64.max 0 (i-1))
    in
    M.tree.construct Y 0
    
  -- Given a tree on a graph, we can compute the node potentials.
  def compute_node_potentials [n] (ts : M.tree.structure[n]) =
    let parents = M.tree.of M.parent_arc ts in
    let cost = M.tree.map
	       (\(arc, orientation) ->
		  (match orientation
		   case #up -> N.neg (M.cost arc)
		   case #down -> M.cost arc)) parents in
    M.tree.top_down_scan (N.+) ts cost

  def compute_flows [n] (ts: M.tree.structure[n]) (node_potentials: M.tree.data[n] N.t) =
    M.tree.bottom_up_scan (N.+) (N.i32 0) ts node_potentials

  -- def update (current_tree
  ----------------------------------------------
  -- Algorithm is unfinished below this point.


  -- def update[n] (G : M.graph) (current_tree : M.tree.structure[n])
  --   (select_arc : M.graph -> M.arc)  
  --   =
  --   let node_potentials = compute_node_potentials G current_tree in
  --   let flows = compute_flows G current_tree node_potentials in
  --   let entering_arc = select_entering_arc G node_potentials flows in
  --   INCOMPLETE
  --   let leaving_arc = INCOMPLETE in
  --   leaving_arc


  -- def network_simplex[n]
  --   (weighted_digraph : M.t)
  --   (initial_spanning_tree_structure: M.tree.structure[n])
  --   =
    

    
    -- def network_simplex (tree_structure : M.tree)
  -- def network_simplex (G : M.graph) (init : M.spanning_tree) =
  -- We assume we are given a graph, and a spanning tree init through the graph.
  -- update G init


  -- def network_simplex (G : M.graph) (init : M.spanning_tree) =
  -- -- We assume we are given a graph, and a spanning tree init through the graph.
  -- update G init

  -- -- The network simplex algorithm takes a cost matrix as input.
  -- def network_simplex_transport [n][m] (cost_matrix : M.cost_matrix[n][m])
  --  (first_marginal : [n]t) (second_marginal : [n]t) 
  -- 		      =
  --   let graph = graph_of cost_matrix first_marginal second_marginal in
  --   network_simplex graph



}
