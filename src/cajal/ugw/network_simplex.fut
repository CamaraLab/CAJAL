import "tree2"

module type network_simplex_context = {
  module tree : tree
  include WeightedDiGraph -- with node = tree.index

  val num_bits : i32
  val get_bit : i32 -> N.t -> i32
  -- type direction = #up | #down
  -- val parent_arc : node -> (arc , direction)
  -- type node_potentials[k] = tree.data [k] N.t
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
  module N = M.N -- numeric type
  type index = M.tree.index -- our index type will be the same as the given implementation of the tree
  type t = N.t
  type direction = #up | #down
  type ns_tree [n][k] =		-- The full state of a spanning tree consists of
    (M.tree.structure[n], -- the isomorphism type of the graph
     M.tree.data [n] direction, -- the orientation of the arc relative to its parent
     M.tree.data [n] N.t, -- the computed node potentials at each node
     M.tree.data [n] N.t, -- the flow through the edge above each point (oriented from root downward)
     [k]M.edge -- all other edges in the graph which aren't present in the tree
    )
  

  import "lib/github.com/diku-dk/sorts/radix_sort"
  import "lib/github.com/diku-dk/segmented/segmented"

  def xor (x: bool) (y: bool) = bool.(x != y)

  -- | begin-streak b is an array of integer indices
  -- for b, such that (begin-streak b)[i] is the array index
  -- for the first boolean in the contiguous streak of booleans that
  -- i belongs to.
  
  -- (begin-streak b)[i] is always <= i,
  -- with equality iff (b = 0 or b[i-1] != b[i]).
  -- Example:
  -- begin_streak [t, t, t, f, f, f, t, t, f] =
  --              [0, 0, 0, 3, 3, 3, 6, 6, 7]
  def begin_streak [n] (b: [n]bool) : [n]i64 =
    let contiguous_streak_start_flag =
      let b_upshift = spread n b[0] (map (+1) (0..<n)) b :> [n]bool in
      map2 xor b b_upshift
    in
    -- I think this is correct but it seems less readable so I deleted it.
    -- let diffs = ([false] ++ map2 xor (map (\i -> b[i]) (0..<(n-1)))
    -- 	       (map (\i -> b[i+1]) (0..<(n-1)))) :> [n]bool in 
    --
    -- This line exploits the fact that the prefix sum (scan) of the array
    -- [k, 0, 0, 0, 0,...] is
    -- [k, k, k, k, k,...]
    let a = map (\i -> if contiguous_streak_start_flag[i] then i else 0) (iota n) in
    segmented_scan (i64.+) (0) contiguous_streak_start_flag a

  -- Let X = { 0, ...,  n-1} and Y = { 0, ..., m-1 }.
  -- Let G be the "complete bipartite graph" between X and Y,
  -- so there is an edge from every element in X to every element in Y.
  -- In this case we care about the complete *directed* bipartite graph "from" X to Y, that is,
  -- the graph which has an edge from x to y for each pair (x,y), but no
  -- edges going the other way. This will simplify the problem,
  -- and let us avoid thinking about negative demand.

  -- We are interested here in constructing a spanning tree for G,
  -- and we are in particular interested in trees having the property that
  -- no two edges cross, so there are no two edges (i, j) and (i', j')
  -- with the property that i < i' and j > j'. It is easy to see that
  -- this requirement forces us to connect 0_X to 0_Y, and (n-1)_X to (m-1)_Y.

  -- I claim that trees satisfying this property are in
  -- one-to-one correspondence with (n-1,m-1)-shuffles (https://planetmath.org/pqshuffle).
  -- For readability we introduce the set X' = { 0,...,n-2 }, Y' = {0, ... , m-2},
  -- and we think of the element i in X' as the dividing line between i \in X and i+1 \in X.
  -- We identify the disjoint union X' \coprod Y' with the set {0,...,n+m-3} under the correspondence
  -- x' \mapsto x', y' \mapsto n-1+y'.
  -- We identify (n-1, m-1)-shuffles with bijective functions X' \coprod Y' \to {0,..., n+m-2} with the property that
  -- x'_0 < x'_1 -> \sigma(x'_0) < \sigma(x'_1), y'_0 < y'_1 -> \sigma(y'_0) < \sigma(y'_1).
  -- We think of elements of X as corresponding to lower subsets of X' -
  -- the number k \in X is associated to the set { x' \in X' | x' < k } of "cuts" or "divisions" to the left of k;
  -- 0 \in X corresponds to the empty subset of X', and X' itself corresponds to n-1.
  
  -- Now that this notation is introduced, for an (n-1,m-1)-shuffle \sigma : X' \coprod Y' \to {0,...,n+m-2}
  -- we associate to this shuffle the tree which has an edge from n_X to n_Y iff for some natural number k,

  -- \sigma^{-1}({0..<k}) = { x' \in X' | x' < n_X } \cup { y' \in Y' | y' < n_Y }.
  
  -- where the \cup should be a disjoint union.
  -- We call this edge the k-th edge.
  -- In particular, if k = 0 then both sides of this equation are empty,
  -- so we always connect 0_X to 0_Y.
  -- Similarly if k = n+m-2 then one can take n_X = n-1 and n_Y = m-1.

  -- For the sake of satisfying the marginal constraints, we must
  -- consider whether the graph defines a valid transport plan.
  -- Let \mu be a measure on X and \nu a measure on Y, both of equal mass.
  -- Let f_X' : X' -> \mathbb{R} be the cumulative sum 
  -- f_X'(x') = \sum_{i<=x'} \mu[i], similarly f_Y' : Y' -> \mathbb{R}.
  -- We write f : X' \coprod Y' \to \mathbb{R} for f_{X'} \coprod f_{Y'} \to \mathbb{R}.
  
  -- A permutation \sigma : X' \coprod Y \to { 0,...,n+m-3} is compatible with a valid transport plan
  -- if and only if it satisfies f(a) < f(b) -> \sigma(a) < \sigma(b).
  -- It is not required to satisfy the stronger condition (a <= b) -> \sigma(a) <= \sigma(b);
  -- one can have f(a) = f(b) and \sigma(a) > \sigma(b).
  -- However, the presence of a pair of distinct elements a, b with f(a) = f(b) does complicate things,
  -- as it implies the existence of degenerate (constrained) edges in the tree, which
  -- is a problem for the network simplex algorithm.
  -- Following Section 11.6 of "Network flows" (1996) we will impose the additional constraint
  -- that we want the tree constructed by the function to be "strongly feasible",
  -- so every edge with zero flow points toward the root.
  -- This requires us to designate a distinguished root node - we choose 0_X.
  -- Whether an edge is pointing "up" or "down" is also a function of the (n-1,m-1)-shufle.
  -- The 0th edge points down, and for k > 0, the k-th edge points down if
  -- \sigma^{-1}(k-1) \in Y', otherwise the k-th edge points up.

  -- For the time being we will ignore measures which have zero mass, because this isn't currently
  -- relevant to our intended applications and because the problem is easily solved at the end-user
  -- level by filtering out zeros. This basically means that we should make our order into a
  -- total order by adding the additional stipulation that if x' \in X', y' in Y' and f(x') = f(y'),
  -- then x' < y'. This makes the optimal solution unique.

  --| initial_spanning_tree_ot accepts two measures, mu and nu, which are assumed to be
  -- nonnegative and have the same mass. It returns a spanning tree through the
  -- complete bipartite graph from mu to nu, which corresponds to a feasible solution to the
  -- optimal transport problem.
  -- All edges in the tree are from mu to nu.
  -- If nu is strictly positive everywhere, then the tree is strongly feasible.

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
      let X = radix_sort_by_key f (M.num_bits) (M.get_bit) (0..<(n+m)) in -- It's important that radix sort is stable.
      begin_streak (map (\i -> i < n) X) |> map (\i -> i64.max 0 (i-1))
    in
    M.tree.construct Y 0
    
  -- Given a tree on a graph, we can compute the node potentials.
  def compute_node_potentials [n] (ts : M.tree.structure[n]) =
    let parents = M.tree.of M.parent_arc ts in
    let cost = M.tree.map
	       (\(arc, orientation) ->
		  (match orientation
		   case #up -> M.cost arc
		   case #down -> M.N.neg (M.cost arc))) parents in
    M.tree.top_down_scan (N.+) ts cost

  def compute_flows [n] (ts: M.tree.structure[n]) (node_potentials: M.tree.data[n] N.t) =
    M.tree.bottom_up_scan (N.+) (N.i32 0) ts node_potentials

  type with_neutral 'a = #neutral | #val a

  def f_with_neutral 't (f : t -> t -> t) (x : with_neutral t) (y : with_neutral t) : with_neutral t =
    match (x, y)
    case (#val x, #val y) -> #val (f x y)
    case (#neutral, _) -> #neutral
    case (_,#neutral) -> #neutral

  def reduce1 't (f : t-> t->t) (ts :[t]) : with_neutral t =
    reduce (f_with_neutral f) #neutral (map (\t -> #val t)) ts

  def argmin 'a : (f : a -> t) = reduce1 (\x y -> if (f x) N.<= (f y) then x else y)

  def dantzig_elimination_rule [n][k][l] (edges : (M.node, M.node)[n]) (G : M.graph[k])
       -- (current_tree : M.tree.structure[l] )
       (node_potentials : M.tree.data[l]) =
    let c_pi i j = N.(M.cost G i j - M.tree.get i + M.tree.get j)
    let x = argmin (uncurry (M.cost G)) node_potentials in
    match x
    case #val (i, j) -> if N.(M.cost G i j < 0) then #val (i, j) else #neutral
    case #neutral -> #neutral

  -- def minor_iteration [n][k][l] (edges : (M.node, M.node)[n]) (G : M.graph[k] ) (current_tree : M.tree.structure[l] ) =
  -- Scan the array of edges and delete or filter all edges which are ineligible

  -- Of the remaining edges, find the one with the lowest violation
  -- Return the new array and the distinguished edge.

  ----------------------------------------------
  -- Algorithm is unfinished below this point.

  def update[n] (G : M.graph) (current_tree : M.tree.structure[n])
    (select_arc : M.graph -> M.arc)
    =
    let node_potentials = compute_node_potentials G current_tree in
    let flows = compute_flows G current_tree node_potentials in
    let entering_arc = select_entering_arc G node_potentials flows in
    INCOMPLETE
    let leaving_arc = INCOMPLETE in
    leaving_arc


  -- def network_simplex[n]
  --   (weighted_digraph : M.t)
  --   (initial_spanning_tree_structure: M.tree.structure[n])
  --   =
    

    
    -- def network_simplex (tree_structure : M.tree)
  def network_simplex (G : M.graph) (init : M.spanning_tree) =
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
