import "tree2"
import "initial_transport_plan"

module type network_simplex_context = {
  module tree : tree
  include WeightedDiGraph with node = tree.index, edge = (node,node)

  val num_bits : i32
  val get_bit : i32 -> N.t -> i32
}

module network_simplex(M : network_simplex_context) = {
  module N = M.N -- numeric type
  type index = M.tree.index -- our index type will be the same as the given
	         -- implementation of the tree
  type t = N.t

  -- The full state of a spanning tree consists of
  -- the isomorphism type of the graph, coded as
  -- a vector of pointers to parents
  -- the orientation of the arc at a relative to its parent
  -- (the direction at the root node is meaningless)
  -- the computed node potentials at each node
  -- the flow through the edge above each point
  -- (oriented from root downward)
  -- all other edges in the graph which aren't present in the tree
  
  type ns_tree [n][k] = {
      parents : M.tree.structure[n];
      dir : M.tree.data[n] direction;
      depth : M.tree.data[n] i32;
      node_potentials : M.tree.data[n] N.t;
      edge_flows : M.tree.data [n] N.t;
      other_edges : [k]M.edge
  };

  module Initial = initial_transport_plan(N, M.tree)
  --| initial_spanning_tree_ot accepts two measures, mu and nu, which are assumed to be
  -- nonnegative and have the same mass. It returns a spanning tree through the
  -- complete bipartite graph from mu to nu, which corresponds to a feasible solution to
  -- the optimal transport problem.
  -- All edges in the tree are from mu to nu.
  -- If nu is strictly positive everywhere, then the tree is strongly feasible.
  def initial_spanning_tree_ot [n][m] (mu: [n]N.t) (nu: [m]N.t):
    M.tree.structure[(n+m)] = Initial.initial_spanning_tree_ot

  -- Given a tree on a graph, we can compute the node potentials.
  def compute_node_potentials [n] (ts : M.tree.structure[n]) =
    let parents = M.tree.of M.parent_arc ts in
    let cost = M.tree.map
	       (\(arc, orientation) ->
		  (match orientation
		   case #up -> M.cost arc
		   case #down -> M.N.neg (M.cost arc))) parents in
    M.tree.top_down_scan (N.+) ts cost

  def compute_flows [n] (ts: M.tree.structure[n]) (supply: M.tree.data[n] N.t) =
    M.tree.bottom_up_scan (N.+) (N.i32 0) ts supply

  def initial_ns_tree [n][m] (mu: [n]N.t) (nu: [m]N.t) : ns_tree[(n+m)][] =
    let parents, directions = initial_spanning_tree_ot mu nu in
    let node_potentials = compute_node_potentials parents in
    let edge_flows = compute_flows parents node_potentials in
    let depth = M.tree.depth parents in
    let other_edges = tabulate_2d n m (\ i j -> (i,j)) in
    {
     parents, dir = directions, depth, node_potentials, edge_flows, other_edges
    }

  def argmin[n] (f : a -> N.t) (arr : a[n]) =
  -- This is not strictly unital but it's probably okay.
  -- if f i <= f j then i else j
    let g : i64 -> i64 -> N.t =
      \i j -> if f (arr[i]) < f (arr[j]) then i else j
    in
    reduce 0 g arr

  def dantzig_elimination_rule [n][k][l] (edges : (M.node, M.node)[n]) (G : M.graph[k])
       (node_potentials : M.tree.data[l]) =
    let c_pi i j = N.(M.cost G i j -
		      M.tree.get i node_potentials
		      + M.tree.get j node_potentials)
    in 
    argmin (uncurry c_pi) edges
    
  module Update = {

  -- Given indices i0 and j0 coding the new edge to be added,
  -- return:
  -- 1. the index w of the lowest common ancestor of i0, j0 in the tree
  -- 2. the flow capacity which can be increased along the cycle
  -- 3. the index below the edge to be removed
  -- 4. a boolean "above_i" which is true if the index below the edge
  -- to be removed is above i, or false if it is below j.

  -- Absorb some of the boilerplate code and case analysis.
  def get_cycle_info_helper ns_tree (i: index) above_i replacement_edge flow_bound =
    let x = M.tree.get i ns_tree.edge_flows in 
    match M.tree.get i ns_tree.dir case
      #up -> if above_i && N.(x < flow_bound) then (i, above_i, x)
	     else (replacement_edge, above_i, flow_bound)
      #down -> if (!above_i) && N.(x <= flow_bound) then (i, above_i, x)
	     else (replacement_edge, above_i, flow_bound)

  def get_cycle_info ns_tree (i0: index) (j0: index) =
    let initial_flow_bound = N.highest in 
    let initial_replacement_edge = i0 in -- This value should be meaningless/arbitrary.
    let initial_above_i = true in -- This value should be meaningless/arbitrary.
    -- If the algorithm is correct, the outcome of this function should
    -- never depend on initial_above_i or initial_replacement_edge.

    let (i, (flow_bound, replacement_edge))
    = loop (i, (flow_bound, replacement_edge)) =
	(i0, (initial_flow_bound, initial_replacement_edge))
      while
	M.tree.get i ns_tree.depth > M.tree.get j0 ns_tree.depth
      do
	(M.tree.parent ns_tree.parents i,
	 
	 )



    
    let (i, (flow_bound, replacement_edge))
    = loop (i, (flow_bound, replacement_edge)) =
	(i0, (initial_flow_bound, initial_replacement_edge))
      while
	M.tree.get i ns_tree.depth > M.tree.get j0 ns_tree.depth
      do
	

    in
    let (j, (flow_bound, replacement_edge, above_i))
    = loop (j0, (flow_bound, replacement_edge, above_i)) =
	(i0, (flow_bound, replacement_edge, initial_above_i))
      while
	M.tree.get j ns_tree.depth > M.tree.get i ns_tree.depth
      do
	(M.tree.parent ns_tree.structure j,
	 (match M.tree.get j ns_tree.dir case
	    #up -> (flow_bound, replacement_edge, above_i)
	   #down -> if N.(flow_bound >= M.tree.get j ns_tree.edge_flows) then
		      (M.tree.get j ns_tree.edge_flows, j, false)
		    else
		      (flow_bound, replacement_edge, above_i)
	 ))
    in
  -- depth i = depth j.
  let (i, j, flow_bound, replacement_edge, above_i) = 
    loop (i, j, flow_bound, replacement_edge, above_i) =
      (i, j, flow_bound, replacement_edge, above_i)
    while i != j do
    let (flow_bound, replacement_edge, above_i) =
      (match M.tree.get i ns_tree.dir case
	 #up -> (if N.(flow_bound > M.tree.get i ns_tree.edge_flows) then
		   (M.tree.get i edge_flows, i, true)
		 else
		   (flow_bound, replacement_edge, above_i))
	#down -> (flow_bound, replacement_edge, above_i))
    in
    let (flow_bound, replacement_edge, above_i) =
      (match M.tree.get j ns_tree.dir case
	 #up -> (flow_bound, replacement_edge, above_i)
         #down -> if N.(flow_bound >= M.tree.get j ns_tree.edge_flows) then
		    (M.tree.get j ns_tree.edge_flows, j, false)
		  else
		    (flow_bound, replacement_edge, above_i)
      )
    in 
    (
      M.tree.parent ns_tree.parents i,
      M.tree.parent ns_tree.parents j,
      flow_bound,
      replacement_edge,
      above_i
    )
  in
  (i, flow_bound, replacement_edge, above_i)

  ------------------------------------------------------

  -- A lot of code duplication here but not sure that removing
  -- the duplication would really make it more readable.

  def update_tree_and_reverse (parents: *M.tree.structure[])
       (dir: *M.tree.data[] direction)
       (flows: *M.tree.data[] N.t)
       (i0: index)
       flow_bound
       replacement_edge
       above_i
   =
    let (i, _, _, _, parents, dir, flows) =
    loop (i, child, child_dir, child_flow,
	  parents : (*M.tree.structure[]),
	  dir : (*M.tree.data[] direction),
	  depths: (*M.tree.data[] i32)
	 ) =
      (i0, j0, if above_i then #down else #up, (N.i32 0), parents, dir, flows)
    while child != replacement_edge
    do
    let i_old_parent = M.tree.parent parents i in
    let new_parents = M.tree.set_parent parents i child in
    let i_old_dir = M.tree.get dir i in
    let new_dir = M.tree.set dir i in
    let i_old_flow = M.tree.get flows i in
    let new_flows = M.tree.set flows i
         match child_dir case
	   #up ->  N.(if above_i then child_flow - flow_bound
		      else child_flow + flow_bound)
           #down -> N.(if above_i then child_flow + flow_bound
		       else child_flow - flow_bound)
    in
    (i_old_parent, i, i_old_dir, i_old_flow, {
    {parents = new_parents, dir = new_dir,
    potentials = potentials, depths = depths, flows = new_flows}
     })
    in
    (i, parents, dir, flows)

  def update_flows_only (parents: M.tree.structure[])
       (dir: M.tree.data[] direction)
       (flows: *M.tree.data[] N.t)
       (i0: index)
       flow_bound
       w
       above_i = 
      while i != w do
      let new_flows = M.tree.set flows i 
	match M.tree.get dir i case
	  #up -> if above_i then N.(M.tree.get flows i - flow_bound)
		 else N.(M.tree.get flows i + flow_bound)
	#down -> if above_i then N.(M.tree.get flows i + flow_bound)
		 else N.(M.tree.get flows i - flow_bound)
      in
      (M.tree.get_parent parents i, new_flows)
    in
    (parents, dir, flows)

  def update_parents_dir_flows
       (parents: *M.tree.structure[])
       (dir: *M.tree.data[] direction)
       (flows: *M.tree.data[] N.t)
       (i0: index) (j0: index) 
       w flow_bound replacement_edge above_i = 
    if above_i then
    let (i, parents, dir, flows) =
      update_tree_and_reverse parents dir flows i0 flow_bound replacement_edge true in
    let (parents, dir, flows) =
      update_flows_only parents dir flows i flow_bound w true in
    let (parents, dir, flows) =
      update_flows_only parents dir flows j0 flow_bound w false in
    else
    let (j, parents, dir, flows) =
      update_tree_and_reverse parents dir flows j0 flow_bound replacement_edge above_i in
    let (parents, dir, flows) =
      update_flows_only parents dir flows j flow_bound w above_i in
    let (parents, dir, flows) =
      update_flows_only parents dir flows i0 flow_bound w (not above_i) in
    (parents, dir, flows)


    -- def update_tree_above_i
    --    (parents: *M.tree.structure[])
    --    (dir: *M.tree.data[] direction)
    --    (flows: *M.tree.data[] N.t)
    --    (i0: index) w flow_bound replacement_edge =

    -- let (i, _, _, _, parents, dir, depths)= 
    -- loop (i, child, child_dir, child_flow,
    -- 	  parents : (*M.tree.structure[]),
    -- 	  dir : (*M.tree.data[] direction),
    -- 	  depths: (*M.tree.data[] i32)
    -- 	 ) =
    --   (i0, j0, #down, (N.i32 0), parents, dir, flows)
    -- while child != replacement_edge
    -- do
    -- let i_old_parent = M.tree.parent parents i in
    -- let new_parents = M.tree.set_parent parents i child in
    -- let i_old_dir = M.tree.get dir i in
    -- let new_dir = M.tree.set dir i in
    -- let i_old_flow = M.tree.get flows i in
    -- let new_flows = match child_dir case
    -- 		      #up -> M.tree.set flows i N.(child_flow - flow_bound)
    -- 		      #down -> M.tree.set flows i N.(child_flow + flow_bound)
    -- in
    -- (i_old_parent, i, i_old_dir, i_old_flow, {
    -- {parents = new_parents, dir = new_dir,
    -- potentials = potentials, depths = depths, flows = new_flows}
    --  })
    -- in
    -- let (_, flows) = 
    --   loop (i, flows) = (i, flows)
    --   while i != w do
    --   let new_flows = match M.tree.get dir i case
    -- 	#up -> M.tree.set flows i N.(child_flow - flow_bound)
    -- 	#down -> M.tree.set flows i N.(child_flow + flow_bound)
    --   in
    --   (M.tree.get_parent parents i, new_flows)
    -- in
    -- (parents, dir, flows)

  }
}
