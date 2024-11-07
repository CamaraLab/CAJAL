module quadratic_gradient_descent (R : real) = {
  import "linear_ot"
  import "common"
  module l = linear_ot R    
  -- Given matrices M1 and M2 of the same shape, return "true" if
  -- | M1[i,j] - M2[i,j] | > tol for some i, j.
  -- Otherwise, return false.
  def errbool (M1 : [][]R.t) M2 tol =
    let err_bool row1 row2 = any (\x -> (R.abs x) R.> tol) (map2 (R.-) row1 row2) in
    any (\x -> x) (map2 err_bool M1 M2)

 -- This is a generic quadratic gradient descent problem which takes as arguments:
 -- + qcf, a "quadratic cost function" operator
 --   which accepts as argument the current optimal transport plan
 --   and returns the cost matrix for the next optimization step
 -- + mu and nu, the marginal distributions along the first and second component
 --   (mu = sums of rows, nu = sums of columns)
 -- + init, the initial best guess for the transport plan
 -- + ct, which controls how many iterations the inner sinkhorn loop runs
 --   before it's interrupted to check termination conditions and normalize the 
 --   matrix entries to safe values
 -- + params, which controls the behavior of the inner sinkhorn algorithm
 --   (see the sinkhorn documentation for discussion)
 -- + tol_outerloop, which controls when to break out of the loop
 
  -- Quadratic gradient descent
  def qgd [n][m] (qcf : [n][m]R.t -> [n][m]R.t) (mu : [n]R.t) (nu : [m]R.t)
             (init : [n][m]R.t) (params : l.params) (tol_outerloop : R.t)
             -- (tol_innerloop : R.t) 
             -- (safe_for_exp_thresh : R.t)
    =
    let step p = l.sinkhorn (qcf p) mu nu params in
    (loop (p0, p1, n) = (init, step init, 0:i32)
    while (errbool p0 p1 tol_outerloop) do
       (p1, l.sinkhorn (qcf p1) mu nu params, n+1))
    |> (\(_,d, n) -> (d, n+1))

  -- Quadratic gradient descent with carried guesses
  -- (i.e., u and v are passed from the last instance)
  module L = linear_ot_generic R
  open L

  def qgd_carry [n][m] (qcf : [n][m]R.t -> [n][m]R.t) (mu : [n]R.t) (nu : [m]R.t)
             (init : [n][m]R.t)
             (eps : R.t)
             (eps_decay : R.t)
             (p : L.G.stability_params)
             (final_tol : R.t)
             (outer_loop_count : i32) =
    let continue a_old a_new _ = l.c.ratio_err a_old a_new R.> p.tol_sinkhorn  in
    let md : G.marginal_distributions[n][m] =
      { dx = replicate n l.c.one, dy = replicate m l.c.one}
    let rp : S.reg_params[n][m] = { epsilon = eps, mu, nu } in
    let C0 = qcf init in
    let (D, u0, v0) = l.c.safe_for_exp C0 (p.safe_for_exp R.* eps) in
    let (u1, v1, P1) = L.G.sinkhorn rp md p u0 v0 D continue in
    let step (u : [n]R.t, v : [m]R.t, plan : [n][m]R.t) =
      L.G.sinkhorn rp md p u v (qcf plan) continue
    in
    (loop (u', v', p', epsilon: R.t) = (u1, v1, P1, eps)
     for _j < outer_loop_count do
     -- let (u1,v1,p1) = L.G.sinkhorn (rp with epsilon = epsilon) md p u' v' (qcf p') continue
     let (u1, v1, p1) = step (u', v', p')
     in
     (u1, v1, p1, epsilon R.* eps_decay))
       -- step (u', v', p')
       |> (\( u,v,plan,eps ) ->
	     L.G.sinkhorn (rp with epsilon = eps) md
			  (p with tol_sinkhorn = final_tol)
			  u v (qcf plan)
			  (\x y _ -> l.c.ratio_err x y R.> final_tol))
       |>  (\( _,_,plan) -> plan)

  def qgd_carry_test_1 [n][m] (qcf : [n][m]R.t -> [n][m]R.t) (mu : [n]R.t) (nu : [m]R.t)
             (init : [n][m]R.t)
             (eps : R.t)
             (p : L.G.stability_params)
             (outer_count : i32) =
    let continue _ _ k = k < outer_count in
    let md : G.marginal_distributions[n][m] =
      { dx = replicate n l.c.one, dy = replicate m l.c.one}
    let rp : S.reg_params[n][m] = { epsilon = eps, mu, nu } in
    let C0 = qcf init in
    let (D, u0, v0) = l.c.safe_for_exp C0 (p.safe_for_exp) in
    L.G.sinkhorn rp md p u0 v0 D continue

}

module fused_gromov_wasserstein (R : real) = {
  import "linear_ot"
  import "common"
  import "gw"
  module gw = gromov_wasserstein R
  open common R
  open quadratic_gradient_descent R
  module l = linear_ot R

  -- The quadratic cost function for Fused Gromov-Wasserstein.
  -- X is the squared pairwise distance matrix for the first space
  -- Y the squared pairwise distance matrix for the second space
  -- P the transport plan

  def fused_gw_qcf M alpha X Y P =
    let B = map (map ( R.* (one R.- alpha))) M in    
    let A = map (map (R.* alpha)) (gw.L2_otimes_T X Y P) in
    map2 (map2 (R.+)) A B

  def fused_gw [n][m] (M: [n][m]R.t) (alpha: R.t)
	       (Ca: [n][n]R.t) (Cb: [m][m]R.t)
               -- (mu: [n]R.t) (nu: [m]R.t)  (init: [n][m]R.t)
               -- (params: l.params) (tol_outerloop: R.t)
    =
    qgd (fused_gw_qcf M alpha Ca Cb)
  -- mu nu init params tol_outerloop

  def qgd_eps_decay [n][m] (qcf : [n][m]R.t -> [n][m]R.t) (mu : [n]R.t) (nu : [m]R.t)
             (init : [n][m]R.t) (params : l.params) (tol_outerloop : R.t) (eps_decay : R.t) =
    (loop (p0, p1, epsilon, n) = (init, l.sinkhorn (qcf init) mu nu params, params.eps, 0:i32)
    while errbool p0 p1 tol_outerloop do
       (p1, l.sinkhorn (qcf p1) mu nu (params with eps = epsilon),
	epsilon R.* eps_decay, n+1))
    |> (\(_,d,_,n) -> (d,n))
    
  def qgd_eps_decay_ct [n][m] (qcf : [n][m]R.t -> [n][m]R.t) (mu : [n]R.t) (nu : [m]R.t)
             (init : [n][m]R.t) (params : l.params) (ct :i32) (eps_decay : R.t)
    =
    (loop (_, p1, epsilon) = (init, l.sinkhorn (qcf init) mu nu params, params.eps)
    for _j < ct do 
       (p1, l.sinkhorn (qcf p1) mu nu (params with eps = epsilon),
	epsilon R.* eps_decay))
    |> (\(_,d,_) -> (d))

  def fused_gw_decay [n][m] (M: [n][m]R.t) (alpha: R.t)
	       (Ca: [n][n]R.t) (Cb: [m][m]R.t) = 
    qgd_eps_decay (fused_gw_qcf M alpha Ca Cb)

  def fused_gw_generic [n][m] (M: [n][m]R.t) (alpha: R.t)
	       (Ca: [n][n]R.t) (Cb: [m][m]R.t) =
    qgd_carry (fused_gw_qcf M alpha Ca Cb)

  def fused_gw_generic_test_1 [n][m] (M: [n][m]R.t) (alpha: R.t) (Ca: [n][n]R.t) (Cb: [m][m]R.t)
              (eps : R.t)
              (inner_loop_count: i32) (outer_loop_count: i32)
      = 
    let (mu : [n]R.t) = replicate n (l.c.one R./ (R.i64 n)) in
    let (nu : [m]R.t) = replicate m (l.c.one R./ (R.i64 m)) in
    let init = l.c.tensor mu nu in
    let p = { exp_absorb_cutoff = (R.f64 1e30), tol_sinkhorn = (R.f64 1e-3),
	      loop_count = inner_loop_count,
	      safe_for_exp = (R.f64 30.0) } in
    qgd_carry_test_1 (fused_gw_qcf M alpha Ca Cb) mu nu init eps p outer_loop_count

}

module fgw64 = fused_gromov_wasserstein f64
entry fused_gw M alpha Ca Cb mu nu init tol_inner eps safe_for_exp_thresh ct
                                tol_outerloop =
  let (p,n) = fgw64.fused_gw M alpha Ca Cb mu nu init
			 { tol = tol_inner, eps, safe_for_exp_thresh, ct} tol_outerloop
  in
  (p , fgw64.frobenius (fgw64.fused_gw_qcf M alpha Ca Cb p) p, n)

entry fused_gw_generic M alpha Ca Cb mu nu init tol_sinkhorn (tol_final : f64) eps eps_decay safe_for_exp
                  exp_absorb_cutoff inner_loop_count (outer_loop_count : i32) =
  fgw64.fused_gw_generic M alpha Ca Cb mu nu init eps eps_decay
			 { exp_absorb_cutoff, loop_count = inner_loop_count,
			   safe_for_exp, tol_sinkhorn} tol_final outer_loop_count 
  |> (\p -> (p , fgw64.frobenius (fgw64.fused_gw_qcf M alpha Ca Cb p) p))

entry fused_gw_generic_test_1 = fgw64.fused_gw_generic_test_1

entry fgw_cost_matrix M alpha Ca Cb p = fgw64.fused_gw_qcf M alpha Ca Cb p
