-- Entropically regularized linear transport
-- in the case where we want marginals to be as close to perfect as is possible
-- (i.e., no explicit rho penalty for Kullback-Leibler divergence.)
-- Aka "Balanced" optimal transport.

import "scaling_generic"
module balanced_linear_ot (M : real) = {
  type t = M.t
  type reg_params [n][m] = { epsilon : t, mu : [n]M.t, nu : [m]M.t }
  def eps (rp : reg_params[][]) = rp.epsilon
  def proxdiv_F1 (rp : reg_params[][]) Kbdx _u =
    map2 (M./) rp.mu Kbdx
  def proxdiv_F2 (rp : reg_params[][]) KTady _v =
    map2 (M./) rp.nu KTady
}

-- : scaling_problem with t = M.t

module linear_ot_generic (M : real) = {
  module S = balanced_linear_ot M
  module G = scaling_generic M S
}

module linear_ot (M : real) = {

  -- import "scaling_unbalanced"
  -- module su = scaling_unbalanced (M)
  import "common"
  module c = common M

  type params = { tol : M.t, eps : M.t, safe_for_exp_thresh : M.t, ct : i32 }
  def entropic_cost C P eps = c.frobenius C P M.+ (eps M.* (c.H P))

  def dykstra_step [n][m] (K : [n][m]M.t) (mu : [n]M.t) (b : [m]M.t) : [n]M.t =
    (map2 (M./) mu (c.mat_vec K b))

  let rescale [m] tol (a : M.t) (k : [m]M.t) =
    if a M.> tol || 
       M.(a < (i32 1 / tol)) then
       let x = M.sqrt a in
       (x, map (M.* x) k)
    else (a, k)

  let rescale_matrix [n][m] tol (a :[n]M.t) (K : [n][m]M.t) (mu : [n]M.t) (nu : [m]M.t) =
    let b0 = dykstra_step (transpose K) nu a in
    let (b1, k2) = unzip (map2 (rescale tol) b0 (transpose K)) in
    let (a', k3) = (dykstra_step (transpose k2) mu b1, transpose k2) in
    unzip (map2 (rescale tol) a' k3)

  def kbar_of C eps safe_for_exp_thresh =
    let Cbar = map (map M.((/ (neg eps)))) C in
    let (Cbar', _, _) = c.safe_for_exp Cbar safe_for_exp_thresh in
    let _ = map M.maximum Cbar' in
    let _ = map M.maximum (transpose Cbar') in
    let _ = Cbar' in
    map (map M.exp) Cbar'

  -- The "balanced" sinkhorn algorithm for optimal transport.
  -- This can be viewed as a special case of the generic unbalanced
  -- optimal transport algorithm but we are writing them separately
  -- write now before we figure out how to combine them.
  -- + C: The cost matrix of the optimal transport problem.
  -- + mu, nu : the marginals that the transport plan is required to satisfy
  --   along the first and second projections (mu = sums of rows, nu = sums of columns)
  -- + p.tol : This determines the exit condition, when successive values of
  --   the row-rescaling vector have ratios less than p.tol, the loop exits.
  -- + p.eps : The regularization parameter, generally when epsilon is larger
  --   the loop will terminate faster but the result will be less accurate
  --   (the transport plan will be farther from the optimal one)
  --   and as epsilon approaches zero the transport cost should
  --   approach the true global minimum. However, as epsilon goes to zero,
  --   there will eventually be numerical instability issues which cause the algorithm
  --   to fail and return nan.
  -- + p.safe_for_exp_thresh - This is a minor parameter which basically sets
  --   an upper bound on how large a number x can be, such that it's still
  --   safe to compute e^x without overflow. It will differ on 32 bit vs 64 bit systems.
  --   I usually just set this to 30.0, because e^30.0 is very far from the upper bound,
  --   but still large enough that it reduces the degree to which the function e^x
  --   tends to trivialize the problem by forcing small numbers toward zero because of underflow.
  -- + p.ct -- The algorithm contains a renormalization step every ct iterations
  --   which moderates extreme values toward zero. Making ct 10-20 will make the algorithm
  --   somewhat faster than if ct = 0 or 1, but will make it numerically less stable.
  --   Currently there is no benchmarking on this.

  def sinkhorn [n][m] (C : [n][m]M.t) mu nu
               (p : params) =
    let Kbar = kbar_of C p.eps p.safe_for_exp_thresh in
    (loop (a0, a1, K) =
       (replicate n c.one, map2 (M./) mu (map M.sum Kbar), Kbar)
     while (c.ratio_err a0 a1) M.> p.tol do
     let a2 = (loop (ai :[n]M.t) = (a1) for _i < p.ct do
	let bi = dykstra_step (transpose K) nu ai in
	dykstra_step K mu bi
	      ) in
     let (a', K') = rescale_matrix (M.f64 1e30) a2 K mu nu in
     (a1, a', K')
    )
    |>
    (\(_, a, K) ->
       let b = (dykstra_step (transpose K) nu a) in
       let P = (c.plan_of a b K) in
       P)

  -- Ok, this is about to get really confusing lmao but I'm about to 
  -- change conventions in a way I hope doesn't come back to bite me later.
  -- Given that epsilon is going to decay as the number of iterations of the
  -- argument progress, it makes sense not to absorb epsilon into u and v?
  -- I think?
  -- So our convention will be that
  -- Kbar = exp[ (-C + u + v)/eps ].

  -- def sinkhorn_w_initial_guess [n][m] (C : [n][m]M.t) mu nu (p : params) (u : [n]M.t)
  --        (v : [m]M.t) =
  --   let Kbar =
  --     let u_over_eps = map (M./ p.eps) u in
  --     let v_over_eps = map (M./ p.eps) v in
  --     map (map (M./ (M.neg p.eps))) C
  --     |> (\mat -> map2 (c.add_vs) mat u_over_eps)
  --     |> transpose
  --     |> (\mat -> map2 (c.add_vs) mat v_over_eps)
  --     |> transpose
  --     |> map (map M.exp)
  --   in
  -- loop (a0, a1, K) =
  --      (replicate n c.one, map2 (M./) mu (map M.sum Kbar), Kbar)
  -- while (c.ratio_err a0 a1) M.> p.tol do

}

module coo_matrix = {
  -- Values, column indices, row indices.
  -- Format follows the wikipedia page.
  -- nnz is the number of nonzero entries.
  -- nrows is the height of the matrix
  -- The last entry of row_indices should be nnz, marking the end of the array.
  import "lib/github.com/diku-dk/segmented/segmented"
  type structure_data [nnz] 'a =
    { nrows: i64,
      ncols: i64,
      row_indices: [nnz]i64,
      col_indices: [nnz]i64
    }

  def reduce_along_rows [nnz]'t 
    (op: t -> t -> t) (ne : t)
    (coo_sd:structure_data[nnz] t) (vals: [nnz]t) : *[]t= 
    let { nrows, ncols = _ , row_indices, col_indices = _ } = coo_sd in
    hist op ne nrows row_indices vals

}


module separable_kernel_coo (M : real) = {
  import "common"
  open common M
  module coo = coo_matrix

  -- def sq_distance (coo_sd : coo.structure_data[])
  --     (x: i64) (delta_X: M.t) (y: i64) (delta_Y: M.t) =
  --   INCOMPLETE

  def cost [nnz] (coo_sd: coo.structure_data[nnz] M.t) (A: [nnz]M.t) (B: [nnz]M.t)
           (K1: i64 -> i64 -> M.t) (K2: i64 -> i64 -> M.t) 
           -- (cost_function:)
	   =
    (tabulate_2d nnz nnz (\i j ->
		  let x1 = coo_sd.row_indices[i] in
		  let x2 = coo_sd.row_indices[j] in
		  let y1 = coo_sd.col_indices[i] in
		  let y2 = coo_sd.col_indices[j] in
		  M.(
		  A[i] M.* (K1 x1 x2) M.* (K2 y1 y2) M.* B[j]
		  M.* (sq ((i64 x2) - (i64 x1)) M.+ (sq (i64 y2 - i64 y1)))
	       )))
    |> map M.sum |> M.sum

  def sk_lookup_table (nrows: i64) (delta_X: M.t) (ncols: i64) (delta_Y: M.t) (epsilon : M.t): ([nrows]M.t, [ncols]M.t)
	       =
    let sq x = x M.* x in
    -- Precompute the table of exponentials for XY coordinates.
    let t1 = tabulate nrows M.(\a -> exp(neg((sq(delta_X * (i64 a))) / epsilon))) in
    let t2 = tabulate ncols M.(\a -> exp(neg((sq(delta_Y * (i64 a))) / epsilon))) in
    (t1, t2)

  -- Compute a vector COO.
  def separable_kernel_multiply_sparse [nnz] (K1 : i64 -> i64 -> M.t)
      (K2 : i64 -> i64 -> M.t) (coo_sd: coo.structure_data[nnz] M.t) (B : [nnz]M.t)
    : [nnz]M.t
    =
    let { nrows, ncols, row_indices, col_indices } = coo_sd in
    let T : [nrows][ncols]M.t =
      let Tt (l : i64) : [nrows]M.t =
	let K_jl = map (\j -> K2 j l) col_indices in
  	let Kjl_Bij = map2 (M.*) K_jl B in
	((coo.reduce_along_rows (M.+) (M.i32 0)
			      coo_sd Kjl_Bij) :> [nrows]M.t)
      in tabulate ncols Tt |> transpose
    in
    map2 (\k l ->
	    let v1 = tabulate nrows (\i -> (K1 i k)) in
	    let v2 = (transpose T)[l] in
	    dot v1 v2) row_indices col_indices

  def sinkhorn_parallel[ncells][nnz]
     (nrows: i64)
     (row_indices: [nnz]i64)
     (ncols: i64)
     (col_indices: [nnz]i64)
     (mu : [nnz]M.t)
     (epsilon : M.t)
     (ct: i64)
     (nu: [ncells][nnz]M.t)
	       =
    let coo_sd = { nrows, ncols, row_indices, col_indices } in
    let (t1, t2) = sk_lookup_table nrows (M.i32 1) ncols (M.i32 1) epsilon
    in
    let K1 i k = t1[i64.abs (k - i)] in
    let K2 j l = t2[i64.abs (j - l)] in

    let update_step B nu =
      let A : [nnz]M.t =
	let pi_X_KB = separable_kernel_multiply_sparse K1 K2 coo_sd B in
	map2 (M./) mu pi_X_KB in
      -- In theory, this next line should be K1^T, K2^T, but
      -- K1 and K2 are symmetric.
      let pi_Y_KTA = separable_kernel_multiply_sparse K1 K2 coo_sd A in
      map2 (M./) nu pi_Y_KTA
    in

    let B0 = replicate ncells (replicate nnz one) in
    let Bfinal =
      loop B' = map2 update_step B0 nu for _i < ct do
	map2 update_step B' nu
    in
    let Afinal =
      let pi_X_KB = map (separable_kernel_multiply_sparse K1 K2 coo_sd) Bfinal in
      map (map2 (M./) mu) pi_X_KB in
    let error = 
      let pi_Y_KTA = map (separable_kernel_multiply_sparse K1 K2 coo_sd) Afinal in
      map2 ratio_err (map2 odot pi_Y_KTA Bfinal) nu
    in
    ( map2 (\A B -> cost coo_sd A B K1 K2) Afinal Bfinal
    , error)
}

module csr_matrix (R : { type t }) = {
  -- Values, column indices, row indices.
  -- Format follows the wikipedia page.
  -- nnz is the number of nonzero entries.
  -- nrows is the height of the matrix
  -- The last entry of row_indices should be nnz, marking the end of the array.
  import "lib/github.com/diku-dk/segmented/segmented"
  open R
  type structure_data [nnz][s] = { ncols: i64, row_indices: [s]i64,
					   col_indices: [nnz]i64 }
  -- vals: [nnz]R.t

  def reduce_along_rows [nnz][s] (op: t -> t -> t) (ne : t)
    (csr_sd:structure_data[nnz][s]) (vals: [nnz]R.t):
              [s-1]t=
    let { ncols = _ , row_indices, col_indices = _} = csr_sd in
    let b = scatter (replicate nnz false) row_indices (replicate s true) in
    let o = segmented_scan op ne b vals in
    tabulate (s-1) (\i -> o[row_indices[i+1]-1])

  -- This is "unsafe."
  -- TODO: Hide this code in general behind an interface.
  def coo_row_indices[nnz][s] (csr_sd: structure_data[nnz][s]): [nnz]i64 =
    let p = tabulate (s-1) (\i -> csr_sd.row_indices[i+1]-csr_sd.row_indices[i]) in
    replicated_iota p :> [nnz]i64

  def create csr_sd (f : i64 -> i64 -> t) =
    map2 f (coo_row_indices csr_sd) csr_sd.col_indices
}


module separable_kernel (M : real) = {
  import "common"
  open common M
  module csr = csr_matrix ({ type t = M.t })

  -- Inputs:
  -- B is the array of values of a CSR matrix whose positions are coded by
  -- the information in csr_sd.
  -- B
  -- K1 and K2 implicitly define a four-dimensional tensor K,
  -- defined by K_ijkl = (K1 i k) * (K2 j l).
  -- Intuitively K is a two-dimensional square matrix of side length nnz;
  -- the entries (i,j), (k,l).
  -- Our goal is to compute the vector Kb,
  -- in time which is less than O(nnz^2).

  def separable_kernel_multiply_sparse [nnz][s] (K1 : i64 -> i64 -> M.t)
      (K2 : i64 -> i64 -> M.t) (B : [nnz]M.t)
      (csr_sd: csr.structure_data[nnz][s])
    =
    let { ncols, row_indices, col_indices = js } = csr_sd in
    let T : [s-1][ncols]M.t =
      let Tt (l : i64) =
	let K_jl = map (\j -> K2 j l) js in
  	let Kjl_Bij = map2 (M.*) K_jl B in
	csr.reduce_along_rows (M.+) (M.i32 0)
			      { ncols, row_indices, col_indices = js } Kjl_Bij 
      in tabulate ncols Tt |> transpose
    in
    csr.create csr_sd
         (\k l -> let a = tabulate (s-1) (\i -> (K1 i k)) in
		  let b = (transpose T)[l] in
		  dot a b)
}

module separable_kernel_ot (M : real) = {
  module SK = separable_kernel M
  open SK
  def cost [nnz][s]
      (csr_sd: csr.structure_data[nnz][s]) (A : [nnz]M.t) (B : [nnz]M.t)
      (cost_function: i64 -> i64 -> M.t)
      (K1: i64 -> i64 -> M.t) (K2: i64 -> i64 -> M.t) =
    let coo = csr.coo_row_indices csr_sd in
    let f i j = (A[i] M.* (K1 coo[i] coo[j])) M.*
		B[j] M.* (K2 csr_sd.col_indices[i] csr_sd.col_indices[j]) M.*
		cost_function i j
    in
    tabulate_2d nnz nnz f |> map M.sum |> M.sum

  def sq_distance (csr_sd : csr.structure_data[][])
      (x: i64) (delta_X: M.t) (y: i64) (delta_Y: M.t)
   =
    let is = csr.coo_row_indices csr_sd in
    (sq ((M.i64 (is[x] - is[y])) M.* delta_X)) M.+
    (sq ((M.i64 ((csr_sd.col_indices)[x] - csr_sd.col_indices[y])) M.* delta_Y))

  def sinkhorn[n][nnz]
     (delta_X: M.t) (delta_Y: M.t) (csr_sd: csr.structure_data[nnz][n])
     (mu : [nnz]M.t) (nu : [nnz]M.t)
     (epsilon : M.t)  (tol : M.t)
	       =
    let sq x = x M.* x in
    -- Precompute the table of exponentials for XY coordinates.
    let t1 = tabulate n M.(\a -> exp(neg((sq(delta_X * (i64 a))) / epsilon))) in
    let t2 = tabulate csr_sd.ncols M.(\a -> exp(neg((sq(delta_Y * (i64 a))) / epsilon))) in
    -- K1 and K2 implicitly define a function K i j k l = (K1 i k) M.* (K2 j l)
    let K1 i k = t1[i64.abs (k - i)] in
    let K2 j l = t2[i64.abs (j - l)] in
    let update_step B =
      let A =
	let pi_X_KB = separable_kernel_multiply_sparse K1 K2 B csr_sd in
	map2 (M./) mu pi_X_KB in
      -- In theory, this next line should be K1^T, K2^T, but
      -- K1 and K2 are symmetric.
      let pi_Y_KTA = separable_kernel_multiply_sparse K1 K2 A csr_sd in
      map2 (M./) nu pi_Y_KTA
    in
    let B0 = replicate nnz one in
    let (_, Bfinal) =
      loop (B, B') = (B0, update_step B0) while
	ratio_err B B' M.> tol do
	(B', update_step B')
    in
    let Afinal =
      let pi_X_KB = separable_kernel_multiply_sparse K1 K2 Bfinal csr_sd in
      map2 (M./) mu pi_X_KB in
    cost csr_sd Afinal Bfinal (\i j -> sq_distance csr_sd i delta_X j delta_Y) K1 K2

  def sinkhorn_ct[n][nnz]
     (delta_X: M.t) (delta_Y: M.t) (csr_sd: csr.structure_data[nnz][n])
     (mu : [nnz]M.t) (nu : [nnz]M.t)
     (epsilon : M.t) (ct: i64)
	       =
    let sq x = x M.* x in
    -- Precompute the table of exponentials for XY coordinates.
    let t1 = tabulate n M.(\a -> exp(neg((sq(delta_X * (i64 a))) / epsilon))) in
    let t2 = tabulate csr_sd.ncols M.(\a -> exp(neg((sq(delta_Y * (i64 a))) / epsilon))) in
    -- K1 and K2 implicitly define a function K i j k l = (K1 i k) M.* (K2 j l)
    let K1 i k = t1[i64.abs (k - i)] in
    let K2 j l = t2[i64.abs (j - l)] in
    let update_step B =
      let A =
	let pi_X_KB = separable_kernel_multiply_sparse K1 K2 B csr_sd in
	map2 (M./) mu pi_X_KB in
      -- In theory, this next line should be K1^T, K2^T, but
      -- K1 and K2 are symmetric.
      let pi_Y_KTA = separable_kernel_multiply_sparse K1 K2 A csr_sd in
      map2 (M./) nu pi_Y_KTA
    in
    let B0 = replicate nnz one in
    let Bfinal =
      loop B' = update_step B0 for _i < ct do
	update_step B'
    in
    let Afinal =
      let pi_X_KB = separable_kernel_multiply_sparse K1 K2 Bfinal csr_sd in
      map2 (M./) mu pi_X_KB in
    let error = 
      let pi_Y_KTA = separable_kernel_multiply_sparse K1 K2 Afinal csr_sd in
      ratio_err (odot pi_Y_KTA Bfinal) nu
    in
    (cost csr_sd Afinal Bfinal (\i j -> sq_distance csr_sd i delta_X j delta_Y) K1 K2,
     error)

  def sinkhorn_parallel[n][nnz][k] (csr_sd: csr.structure_data[nnz][n])
		       (mu: [nnz]M.t) (nu: [k][nnz]M.t) (epsilon: M.t) (ct: i64) =
    map (\nu -> sinkhorn_ct (M.i32 1) (M.i32 1) csr_sd mu nu epsilon ct) nu
}

module SK = separable_kernel_ot (f64)

entry sk_sinkhorn [npts][s] (delta_X: f64) (delta_Y: f64)
		  (col_indices: [npts]i64) (ncols: i64) (row_indices:[s]i64)
     (mu: [npts]f64) (nu: [npts]f64) (epsilon: f64) (tol: f64)
 = SK.sinkhorn delta_X delta_Y {col_indices, ncols, row_indices}
	       mu nu epsilon tol

-- entry sk_sinkhorn_parallel [npts][s][k]

-- 		  (col_indices: [npts]i64) (ncols: i64) (row_indices:[s]i64)
--      (mu: [npts]f64) (nu: [k][npts]f64) (epsilon: f64) (ct: i64)
--  = SK.sinkhorn_parallel {col_indices, ncols, row_indices}
-- 	       mu nu epsilon ct |> unzip

module SK_COO = separable_kernel_coo f64
entry sk_sinkhorn_parallel = SK_COO.sinkhorn_parallel

 --      (nrows: i64) (row_indices: [npts]i64)
 --      (ncols: i64 )(col_indices: [npts]i64) (ncols: i64) (row_indices:[s]i64)
 --     (mu: [npts]f64) (nu: [k][npts]f64) (epsilon: f64) (ct: i64)
 -- = SK.sinkhorn_parallel {col_indices, ncols, row_indices}
 -- 	       mu nu epsilon ct |> unzip

module EB = linear_ot f64
entry entropic_emd C mu nu ct eps safe_for_exp_thresh tol =
  EB.sinkhorn C mu nu {ct, eps, safe_for_exp_thresh, tol}
