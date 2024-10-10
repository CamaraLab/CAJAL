-- Generic utility functions,
-- in particular functionality relevant to entropic methods.

module common (M : real) = {
  -- import "lib/github.com/diku-dk/sorts/radix_sort"
  type t = M.t
  def zero = M.i64 0
  def one = M.i64 1
  def scale (v : []t) (a :t) = map (M.* a) v
  def divv (v : []t) (a : t) = map (M./ a) v
  def add_vs (v : []t) (a :t) = map (M.+ a) v
  def sub_vs (v : []t) (a :t) = map (M.- a) v
  def odot u v= map2 (M.*) u v
  def dot v v' = reduce (M.+ ) zero (odot v v')
  def mat_mul [n][m][k] (A :[n][m]t) (B: [m][k]t) : [n][k]t =
    map (\a -> (map (dot a) (transpose B))) A
  def mat_vec K v = map (dot v) K
  def max_abs v = reduce M.max zero (map M.abs v)
  def err v v' = max_abs (map2 (M.-) v v')
  def ratio_err v v' = map2 (M./) v v' |> map (M.- one) |> max_abs
  def frobenius P Q = map2 dot P Q |> M.sum
  def sum_rows = map M.sum
  def avg [k] (a: [k](M.t)) = M.sum a M./ (M.i64 k)
  def sum_cols m = sum_rows (transpose m)
  def tensor [n] [m] (u : [n]t) (v : [m]t) : [n][m]t = map (scale v) u
  def ext_sum [n] [m] (u : [n]t) (v :[m]t) : [n][m]t = map (add_vs v) u
  def replicate 'a (n : i64) (x : a) : [n]a = map (\_ -> x) (0..<n) 
  def sq x = x M.* x
  -- Given two vectors of points, x and y, compute the matrix of pairwise
  -- *squared* distances d(x_i,y_j) in Euclidean space.
  -- First dimension is number of points, second column is dimension.
  -- Each row is one point.
  def cdist_sq [n][m][k] (x : [n][k]M.t) (y : [m][k]M.t) : [n][m]M.t =
    let sqdist (a : [k]M.t) (b : [k]M.t) =
      let z = map2 (M.-) a b in
      dot z z
    in
    map (\xi -> map (sqdist xi) y) x

  -- First dimension is number of points, second column is dimension.
  -- Each row is one point.
  def pdist [n][k] (x : [n][k]t) : [n][n]t = cdist_sq x x |> map (map M.sqrt)

  def parallel_while_extract 'a 'b 'c [n]
    (update : a -> b -> a) (extract : a -> b -> c) (exit_condition : a -> bool)
    (varying_inputs : [n]a) (constant_inputs : [n]b) : [n]c =
    let (_, outputs) =
    loop ((p, outputs) : ([](i64,a), *[n]c)) =
      (zip (0..<n) varying_inputs, map2 extract varying_inputs constant_inputs)
    while trace (length p) > 0 do
    let (remaining_indices, remaining_varying_inputs) = unzip p in
    let remaining_constants = (map (\i -> constant_inputs[i]) remaining_indices)
    -- Note that update can contain a for loop if desired.    
    let new = map2 update
		   remaining_varying_inputs
		   remaining_constants in
    let finished = map exit_condition new in
    let results = filter (\(a,_) -> a)
			 (zip finished (zip3 remaining_indices new remaining_constants))
		  |> map (\ (_,(i,a, b)) -> (i, extract a b)) in
    let outputs =
      let (is, cs) = unzip results in
      scatter outputs is cs
    in
    let next_inputs : [](i64, a) = filter (\(a,_)-> not a)
			     (zip finished (zip remaining_indices new)) |>
			map (\a -> a.1) in
    (next_inputs, outputs)
    in outputs

  def parallel_while_test =
    let update_fn a (_ : i64) = a M.* a in
    let extract a _ = a in
    let exit_condition a = a M.> (M.i64 100) in
    parallel_while_extract update_fn extract exit_condition


  -- klu is "un-normalized" or "unbiased", note the lack of -1 or + y.
  def klu x y = if x M.== zero then zero else x M.* M.log (x M./ y)
  -- KLu is the "un-normalized" KL divergence. 
  -- KLu(a | b) = \sum_i a_i ln(a_i/b_i)

  -- The partial derivative of klu as a function of the argument x.
  --- klu (x + dx) y \approx klu x y + dx * klu' x y.
  def klu' x y =  M.(i64 1 + (log x - log y))
    
  -- This is klu tweaked by adding 1e-10, which is deemed small.
  def klu_thibsej x y =
    x M.* M.log (x M./ y M.+ M.f64 1e-10) M.- x M.+ y

  def KLu [n] (a : [n]M.t) (b : [n]M.t) =
    M.sum (map2 klu a b)

  -- The "biased" version.
  def kl x y = if x M.== zero then y else x M.* M.log (x M./ y) M.- x M.+ y

  def KL [n] (a : [n]M.t) (b : [n]M.t) =
    M.sum (map2 kl a b)

  -- The partial derivative of KL as a function of the vector a.
  def KL' [n] (a : [n]M.t) (b : [n]M.t) = map2 (\a b -> M.(log (a/b))) a b

  def KL2 [n][m] (p : [n][m]t) (q : [n][m]t) =
    map2 KL p q |> (M.sum)

  -- Identical to KLu2(P | \mu\otimes\nu) but with a different implementation
  -- that might be more performant.
  def KLu2 [n][m] (P : [n][m]t) (mu :[n]t) (nu : [m]t) : t =
    map2 (\mu_i P_i ->
	    (map2 (\nu_j P_ij ->
		     klu P_ij (mu_i M.* nu_j)) nu P_i |> M.sum)) mu P |> M.sum

  -- The gradient vector  \nabla KLu2(P | \mu \otimes \nu)
  -- as a function of P, \mu and \nu being held constant;
  -- KLu2 (P + dP) mu nu \approx KLu2 P mu nu + frobenius (KLu2' P mu nu) dP 
  def nabla_KLu2 [m][n] (P : [m][n]t) (mu :[m]t) (nu : [n]t) : [m][n]t =
    let lg_P_plus_1 = map M.(map (\a -> i64 1 + log a)) P in
    let x = transpose lg_P_plus_1 |> map2 (\a v -> map (M.- a) v) (map M.log nu) |> transpose in
    map2 sub_vs x (map M.log mu)
    
  -- Thibault Sejourne modifies KL by 1e-10 to the log so it doesn't break.
  def KLu2_thibsej [n][m] (P : [n][m]t) (mu :[n]t) (nu : [m]t) : t =
    map2 (\mu_i P_i ->
	    (map2 (\nu_j P_ij ->
		     klu_thibsej
		     P_ij (mu_i M.* nu_j)) nu P_i |> M.sum)) mu P |> M.sum
      
  -- This is the expression from equation 6 in the unbalanced GW
  -- paper. It is the "biased" KL distance
  -- KL(pi \otimes gamma | \mu\otimes nu\otimes \mu \otimes \nu).
  def KL4 pi gamma mu nu =
    let massp = map M.sum pi |> M.sum in
    let massq = map M.sum gamma |> M.sum in
    let m_mnmn =
      let m_mu = M.sum mu in
      let m_nu = M.sum nu in
      let m_mn = m_mu M.* m_nu in
      m_mn M.* m_mn
    in
    (massq M.* KLu2 pi mu nu) M.+
    (massp M.* KLu2 gamma mu nu) M.+
    m_mnmn M.-
    (massp M.* massq) 

  -- This is not identical to KL4, it adds a small fudge term to
  -- avoid having to deal with zeros.
  -- It should be, numerically, reasonably close to KL4.
  -- Added for comparison with the original UGW paper.
  def KL4_fudge pi gamma mu nu =
    let massp = map M.sum pi |> M.sum in
    let massq = map M.sum gamma |> M.sum in
    let m_mnmn =
      let m_mu = M.sum mu in
      let m_nu = M.sum nu in
      let m_mn = m_mu M.* m_nu in
      m_mn M.* m_mn
    in
    (massq M.* KLu2_thibsej pi mu nu) M.+
    (massp M.* KLu2_thibsej gamma mu nu) M.-
    (massp M.* massq) M.+
    m_mnmn

  -- The gradient of KL4(P \otimes P \mid (\mu\otimes \nu)^{\otimes 2})
  def nabla_KL4  [m][n] (P : [m][n]t) (mu :[m]t) (nu : [n]t) : [m][n]t =
    let lgP_mu_nu =
      map (map M.log) P
      |> transpose
      |> map2 (\a v -> (map (M.- a) v)) (map M.log nu)
      |> transpose
      |> map2 (\a v -> (map (M.- a) v)) (map M.log mu)
    in
    let massP = map M.sum P |> M.sum in
    map M.(map (\b -> fma (i64 2 * massP) b (i64 2 * (KLu2 P mu nu)))) lgP_mu_nu
      
  -- def nabla_KL4  [m][n] (P : [m][n]t) (mu :[m]t) (nu : [n]t) : [m][n]t =
  --   let mP = M.sum (map M.sum P) in
  --   let two_nabla_KLuP_times_mP =
  --     let a = M.(mP + mP) in
  --     let c = let klu2_ = (KLu2 P mu nu) in M.(klu2_ + klu2_) in
  --     map (map (\b -> M.fma a b c)) (nabla_KLu2 P mu nu)
  --   in
  --   let b = M.neg (M.i64 2) |> (M.* mP) in
  --   map2 (map2 M.(\a c -> fma a b c)) P two_nabla_KLuP_times_mP
      
  -- The "biased" entropy of a measure.
  def h = (\a -> if a M.== zero then zero else a M.* (M.log a M.- one))
  def H (p : [][]M.t) =
    M.sum (map (\row -> M.sum (map h row)) p)

  def count_nan v = map (\a -> if M.isnan a then 1 else 0) v |> i32.sum
  def count f v = map f v |> map (\b -> if b then 1 else 0) |> i32.sum
  -- Returns (u, v, D) where
  -- C_ij = D_ij - u_i - v_j and
  -- u_i and v_ij are as large as possible such that
  -- no value of D_ij exceeds thres.
  -- Each row and column of D should have maximum thres.

  def count_nan2d m = map count_nan m |> reduce (i32.+) (0:i32)
  def safe_for_exp [n][m] (C : [n][m]t) thres : ([n][m]t,[n]t,[m]t) =
    let helper row = M.minimum (map (thres M.-) row) in
    let u = map helper C in
    let C0 = map2 add_vs C u in
    let v = map helper (transpose C0) in
    let D = map2 add_vs (transpose C0) v |> transpose in
    (D,u,v)

  def plan_of a b K = map2 scale K a |> transpose |> (\z -> map2 scale z b) |> transpose

  -- Gradient is assumed to be negative!!
  -- 0 < decay < 1; the decay parameter tells how much closer to get to the original point
  -- with every iteration.
  -- gradient is  d(loss(point + t * (gradient - guess)))/dt.
  -- c is the acceptable fraction of the "desired" loss based on linear approximation.

  def armijo_line_search [m][n] loss_fn
    initial_loss
    (point : [m][n]M.t) (diff:[m][n]M.t) gradient_fn decay c =
    let gradient =
      let gradient = gradient_fn point diff in
      assert M.((gradient) < zero)
	     gradient
    in
    loop (diff, current_loss, alpha) = (diff,
					-- loss_fn point diff,
					loss_fn (map2 (map2 (M.+)) point diff),
					M.i64 1) while
       let x = (current_loss) M.- initial_loss in
       M.((x) >= neg (f64 1)) &&
       let y = gradient M.* alpha in
       let z =  x M./ y in
       (z) M.<= c && alpha M.> M.f64 1e-50
       -- current_loss M.>= initial_loss M.+ alpha M.* gradient M.* c && alpha M.> M.f64 1e-50
    do
    let alpha' = alpha M.* decay in
    let diff' = map (map (M.* decay)) diff in
    (diff',
     -- loss_fn point diff',
       loss_fn (map2 (map2 (M.+)) point diff'),
       alpha')

  -- This is a very slightly different algorithm which is
  -- meant for the case where one starts with T' rather than \Delta T.
  -- def armijo_line_search' [m][n] loss_fn
  --   initial_loss
  --   (point : [m][n]M.t) (guess : [m][n]M.t) gradient_fn decay c =
  --   let diff = map2 (map2 (M.-)) guess point in
  --   let alpha = c in 
  --   let gradient = gradient_fn point diff in
  --   let step (diff, _, alpha) =
  --     let diff' = map (map (decay M.*)) diff in
  --     (diff',
  --      loss_fn (map2 (map2 (M.+)) point diff'),
  --      alpha M.* c)
  --   in 
  --   if M.(loss_fn guess < initial_loss M.+ gradient M.* alpha) then (diff, loss_fn guess) else
  --   let (diff, loss, _) =
  --     loop (diff, loss, alpha) = step (diff, loss_fn guess, alpha)
  --     while M.(loss >= initial_loss + gradient M.* alpha) do
  -- 	step (diff, loss, alpha)
  --   in
  --   (diff, loss)

}
