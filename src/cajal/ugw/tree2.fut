module type tree = {

  -- | The "structure" of a tree defines its isomorphism type,
  -- and identifies a distinguished root node,
  -- but the nodes in the tree structure don't hold data.
  type structure [n]

  -- | A tree index is a pointer to a node in the graph.
  -- Given a tree index, we can access the node at that point.

  type index

  -- | The tree-structure encodes the isomorphism type of the tree,
  -- and the root.
  type data [n] 'a

  -- | A structure is represented as an array of pointers from each node to its parent.
  -- The pointer for the root node should be itself.
  val construct [n] : [n]i64 -> i64 -> structure[n]
  val get [n] 'a : index -> structure [n] -> data [n] a -> a
  val root [n] : structure [n] -> index
  val is_root[n]  : structure[n] -> index -> bool
  val map [n] 'a 'b : (a -> b) -> data [n] a -> data [n] b
  val parent [n] : structure [n] -> index -> index
  val of [n] 'a : (index -> a) -> structure [n] -> data[n] a
  val map' [n] 'a 'b : (index -> a -> b) -> structure [n] -> data [n] a -> data [n] b

  -- | Inclusive scan of a tree, given an associative operator. Work
  -- *O(n log(n))*, span *O (log(n))*.
  -- This is "top-down scan". The root value of the returned tree
  -- is the same as the root value of the original tree.
  -- The fact that only tree_data is returned implies
  -- that this tree is isomorphic to the original.
  val top_down_scan [n] 'a : (a -> a -> a) ->
    structure [n] -> data [n] a -> data [n] a
  -- | Reduction with an associative *and commutative* operator.
  val reduce_comm [n] 'a : (a -> a -> a) -> a -> data [n] a -> a
  val bottom_up_scan [n] 'a : (a -> a -> a) -> a ->
    structure [n] -> data [n] a -> data [n] a
}

module tree_impl = {
  type structure [n] = { P: [n]i64, root : i64 }
  type index = i64
  type data [n] 'a = [n]a
  def get (i: index) _ data = data[i]
  def root[n] (ts : structure[n]) = ts.root
  def is_root[n] (ts: structure[n]) i = ts.root == i
  def map = map
  def of [n] f (ts: structure[n]) = map f ts.P
  def map' [n] f (a: structure[n]) b = map2 f a.P b
  def parent[n] (ts: structure[n]) (i: index) = ts.P[i]
  def construct P root = { P, root }

  local def wyllie_scan_step [n] 'a (op: a -> a -> a)
                              (V: [n]a) (P: [n]i64) =
    let f i = if P[i] == i
              then (V[i], P[i])
              else (V[i] `op` V[P[i]], P[P[i]])
    in unzip (tabulate n f)

  local def wyllie_scan [n] 'a (op: a -> a -> a)
                         (V: [n]a) (P: [n]i64) =
    let (V,_) = loop (V, P) for _i < 64 - i64.clz n do
                  wyllie_scan_step op V P
    in V

  def top_down_scan[n] 'a (op: a -> a -> a) (ts: structure[n]) (V: data[n] a) = 
    (wyllie_scan op V ts.P)

  def reduce_comm [n] 'a (op: a -> a -> a) (ne: a) (l: data [n] a) =
    reduce_comm op ne l

  local def bottom_up_scan_step [n] 'a (op: a -> a -> a) (ne: a) (root: i64)
        (V : [n]a) (P : [n]i64) =
    let V' = copy V in 
    let Vo = reduce_by_index V' op ne P V in
    (Vo, tabulate n (\i -> if P[i] == root then -1 else P[P[i]]))
  
  def bottom_up_scan [n] 'a (op: a -> a -> a) (ne: a) (l: structure [n])
    (V: data[n] a)
     =
    let {P, root} = l in
    let (V', _) =
      loop (V, P) for _i < 64 - i64.clz n do
	bottom_up_scan_step op ne root V P
    in V'
}

module tree : tree = tree_impl

module Make_tree (tree: tree) = {
  open tree
  
}

module type WeightedDiGraph = {
  module N : numeric
  type node
  -- This type signature implies that each arc either knows its cost
  -- directly or contains sufficient information to compute the cost.
  type arc
  val cost: arc -> N.t
  type t
}



-- module WeightedDiGraph_impl = {
--   module N = i64
--   type node = i64
--   type arc = i64
--   -- type cost_data [n] = i64
--   def cost (i: arc) = i
-- }

-- module WeightedDigraph : WeightedDiGraph = WeightedDiGraph_impl
