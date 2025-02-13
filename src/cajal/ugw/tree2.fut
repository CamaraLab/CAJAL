module type tree = {

  -- | The "structure" of a tree defines its isomorphism type,
  -- and identifies a distinguished root node,
  -- but the nodes in the tree structure don't hold data.
  type structure [n]

  -- | A tree index is a pointer to a node in the graph.
  -- Given a tree index, we can access the node at that point.
  type index

  -- | Data for a tree contains a stored value for each node.
  type data [n] 'a

  -- | A structure is represented as an array of pointers from each node to its parent.
  -- The pointer for the root node should be itself.
  -- The second argument should be the index of the root node.
  val construct [n] : [n]i64 -> i64 -> structure[n]

  -- | Access the data in a node.
  val get [n] 'a : index -> data [n] a -> a

  -- | Get the index of the root node.
  val root [n] : structure [n] -> index

  -- | Check whether the index is the root.
  val is_root[n]  : structure[n] -> index -> bool

  val map [n] 'a 'b : (a -> b) -> data [n] a -> data [n] b
  val parent [n] : structure [n] -> index -> index

  -- | Populate the nodes of a tree from a function of the index.
  val of [n] 'a : (index -> a) -> structure [n] -> data[n] a

  -- | Compute a new tree, where each value in the new node can be both a function
  -- of the value in the original node and its index.
  val map' [n] 'a 'b : (index -> a -> b) -> structure [n] -> data [n] a -> data [n] b

  -- | Inclusive scan of a tree, given an associative operator. Work
  -- *O(n log(n))*, span *O (log(n))*.
  -- This is "top-down scan". The root value of the returned tree
  -- is the same as the root value of the original tree.
  -- The fact that only tree_data is returned is because
  -- this tree is isomorphic to the original, so it has the same structure.
  val top_down_scan [n] 'a : (a -> a -> a) ->
    structure [n] -> data [n] a -> data [n] a

  -- | Reduction with an associative *and commutative* operator.
  val reduce_comm [n] 'a : (a -> a -> a) -> a -> data [n] a -> a

  -- | This is again an inclusive scan, where the leaves in the new
  -- tree should all have value equal to their original value.
  val bottom_up_scan [n] 'a : (a -> a -> a) -> a ->
    structure [n] -> data [n] a -> data [n] a

  -- | A vector of node depths starting at 1.
  -- If it is more convenient to start indexing at 0
  -- then you can just map (\x -> x-1)
  -- across the node depths.
  val depth [n] : structure [n] -> data [n] i32
}

module tree_impl = {
  type structure [n] = { P: [n]i64, root : i64 }
  type index = i64
  type data [n] 'a = [n]a
  def get (i: index) data = data[i]
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

  -- The use of -1 as a special value here is not meant to function as the
  -- special flag value used to mark the parent of the root.
  -- In our coding of trees, a root's parent is itself.
  -- Instead, the -1 is an arbitrary index lying outside the array,
  -- which to the reduce_by_index function represents
  -- that this value should not get added to the output array.
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

  def depth [n] (structure: structure[n]) : data[n] i32 =
    let const1 = of (\_ -> 1) structure in
    top_down_scan (i32.+) structure const1
}

module tree : tree = tree_impl

module type WeightedDiGraph = {
  module N : numeric
  type node_index
  type node
  type edge
  val source : edge -> node_index
  val target : edge -> node_index
  type graph [k]
  val get[k] : graph[k] -> node_index -> node
  val cost[k] : graph[k] -> edge -> N.t					 
}

module WeightedDiGraph_Example : WeightedDiGraph = {
  module N = float64
  type node_index = i64
  type node = (float64, float64, float64)
  type edge = (node_index, node_index)
  def source (x, _) = x
  def target (_, y) = y
  type graph[k] = [k]node
  def get[k] (g : graph[k]) (i : node_index) = g[i]
  def cost[k] (g : graph[k]) ((i,j): edge) =
    let (x1, y1, z1) = node[i] in 
    let (x2, y2, z2) = node[i] in
    N.(
      let dX = x1 - x2 in
      let dY = y1 - y2 in
      let dZ = z1 - z2 in
      sqrt (dX * dX + dY * dY + dZ * dZ)
    )
}
