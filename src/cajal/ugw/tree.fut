module type tree = {
  -- | A rooted tree, where each node has a pointer to the parent,
  -- and the parent points to itself.

  type tree [n] 'a
  -- | A partial function extracting the root of the tree.
  -- *O(1)*.
  val root [n] 'a : tree [n] a -> a

  -- | Apply function to every node in the tree.
  val map [n] 'a 'b : (a -> b) -> tree [n] a -> tree [n] b

  -- | Inclusive scan of a tree, given an associative operator. Work
  -- *O(n log(n))*, span *O (log(n))*.
  -- This is "top-down scan". The root value of the returned tree
  -- is the same as the root value of the original tree.
  val scan [n] 'a : (a -> a -> a) -> tree [n] a -> tree [n] a
  -- | Reduction with an associative operator.
  -- val reduce [n] 'a : (a -> a -> a) -> a -> tree [n] a -> a

  -- | Reduction with an associative *and commutative* operator.
  val reduce_comm [n] 'a : (a -> a -> a) -> a -> tree [n] a -> a

  -- | Given a cost function on pairs of nodes, return the
  -- cost along the path to the root.
  val cost_scan [n] 'a 'b : (b -> b -> b) -> (a -> a -> b)
             -> tree [n] a -> tree [n] b
  val bottom_up_scan [n] 'a : (a -> a -> a) -> a -> tree [n] a -> tree [n] a
}

module tree : tree = {
  -- S[root] = root

  type tree [n] 'a = { pred: [n]i64
                     , V: [n]a
                     , root: i64
                     }

  -- The root value.
  def root [n] 'a (t: tree [n] a) = assert (n > 0) t.V[t.root]

  def wyllie_scan_step [n] 'a (op: a -> a -> a)
                              (V: [n]a) (S: [n]i64) =
    let f i = if S[i] == i
              then (V[i], S[i])
              else (V[i] `op` V[S[i]], S[S[i]])
    in unzip (tabulate n f)

  def wyllie_scan [n] 'a (op: a -> a -> a)
                         (V: [n]a) (S: [n]i64) =
    let (V,_) = loop (V, S) for _i < 64 - i64.clz n do
                  wyllie_scan_step op V S
    in V

  def cost_tree [n] 'a 'b (cost : a -> a -> b) (V : [n]a) (S :[n]i64) =
   tabulate n (\i -> cost V[i] V[S[i]])

  def cost_scan [n] 'a 'b (op: b -> b -> b) (cost: a -> a -> b) (t: tree [n] a) =
    let {pred, V, root} = t in
    let dists = cost_tree cost V pred in
    let V' = wyllie_scan op dists pred in
    {pred, V=V', root}

  def depth [n] (S: [n]i64) : [n]i64 = wyllie_scan (+) (replicate n 1) S

  def to_array [n] 'a (l: tree [n] a) =
    scatter (copy l.V) (map (\i -> n - i) (depth l.pred)) l.V

  def scan [n] 'a (op: a -> a -> a) (l: tree [n] a) =
    l with V = (wyllie_scan op l.V l.pred)

  def reduce_comm [n] 'a (op: a -> a -> a) (ne: a) (l: tree [n] a) =
    reduce_comm op ne l.V

  def flow_list 'a 'b (f : a -> a -> b) {S, V, last, head} =
    { S, V = map (\(i: i64) -> f V[i] V[S[i]]) S, last, head}

  def map [n] 'a 'b (f: a -> b) (l: tree [n] a) : tree [n] b =
    {pred = l.pred, V = map f l.V, root = l.root}

  def bottom_up_scan_step [n] 'a (op: a -> a -> a) (ne: a) (root: i64)
        (V : [n]a) (P : [n]i64) =
    let V' = copy V in 
    let Vo = reduce_by_index V' op ne P V in
    (Vo, tabulate n (\i -> if P[i] == root then -1 else P[P[i]]))
  
  def bottom_up_scan [n] 'a (op: a -> a -> a) (ne: a) (l: tree [n] a) =
    let {pred, V, root} = l in
    let (V', _) =
      loop (V, pred) for _i < 64 - i64.clz n do
	bottom_up_scan_step op ne root V pred
    in {pred, V=V', root}

}
