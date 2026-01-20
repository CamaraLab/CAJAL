-- This file implements algorithms from the paper "Scaling algorithms
-- for unbalanced optimal transport"
-- by CHIZAT, GABRIEL PEYR´E, BERNHARD SCHMITZER,
-- AND FRANCOIS-XAVIER VIALARD
-- and the paper
-- "The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation"
-- by Thibault Séjourné, François-Xavier Vialard and
-- Gabriel Peyré
-- From page 22.
-- This file contains the mechanisms that are usable
-- in the balanced and unbalanced setting.

module type scaling_problem = {
  -- Type of numbers.
  type t
  -- Regularization parameter record type.
  -- For an example see scaling_unbalanced.fut
  
  type reg_params [n][m]
  val eps [n][m] : reg_params[n][m] -> t
  -- Get a from Kb and u
  -- (really, get abar from K bbar and u)
  val proxdiv_F1 [n][m] : reg_params[n][m] -> [n]t -> [n]t -> [n]t
  -- Get b from K^T a.
  -- (bbar from K^T abar)
  val proxdiv_F2 [n][m] : reg_params[n][m]-> [m]t -> [m]t -> [m]t
}

module scaling_generic (M : real) (S : scaling_problem with t = M.t) = {
  import "common"

  open common M

  def rescale [m] upper_tol lower_tol ai ui ki : (M.t, M.t, [m]M.t)=
    if (ai M.>= upper_tol) || (ai M.< lower_tol) then
    let lgai = M.log ai in
    let output = M.((sqrt ai, ui + (lgai / i32 2), map (* (sqrt ai)) ki)) in
	      output 
    else
      (ai, ui, ki)
      
  def matrix_rescale upper_tol lower_tol a u k =
    let arr = map3 (rescale upper_tol lower_tol) a u k in
    unzip3 arr

  -- This is the main loop of algo3 and algo2, extracted for modularity.
  -- Our convention here is that Kbar[i][j]  == (exp ubar[i]) * K[i][j] * (exp vbar[j]).
  -- The input values of abar and bbar can be used to represent the best guess of the transport plan
  -- to use as a starting point.
  -- In the absence of a reasonable initial guess, we can just set them to one.
  -- def algo3_core_loop [n][m] (r : otp[n][m])
  type algo3_state [n][m] = { abar : [n]t, ubar :[n]t, bbar : [m]t, vbar :[m]t, Kbar : [n][m]t }

  def plan [n][m] (s : algo3_state[n][m]) =
    map2 scale s.Kbar s.abar |> transpose
    |> (\a -> map2 scale a s.bbar) |> transpose

  def normalize_state [n][m] (st : algo3_state[n][m]) (exp_absorb_cutoff :t) :
    algo3_state[n][m]=
    let (abar_n, ubar_n, kbar_n0) =
      if any (M.>= exp_absorb_cutoff) st.abar ||
	 any (M.<= M.recip exp_absorb_cutoff) st.abar then
	matrix_rescale exp_absorb_cutoff (M.recip exp_absorb_cutoff) st.abar st.ubar st.Kbar
      -- (|> manifest)
      else
      (st.abar, st.ubar, st.Kbar)
      -- (|> manifest)
      in
    let (bbar_n, vbar_n, kbar_n) = 
      if any (M.>= exp_absorb_cutoff) st.bbar ||
	 any (M.<= M.recip exp_absorb_cutoff) st.bbar then
      let (bbar2, vbar2, kbar2t) =
	matrix_rescale exp_absorb_cutoff (M.recip exp_absorb_cutoff) st.bbar st.vbar
		       (transpose kbar_n0)
      in
      (bbar2, vbar2, transpose kbar2t)
      else
	(st.bbar, st.vbar, kbar_n0)
    in
    { abar = abar_n, ubar = ubar_n, Kbar = kbar_n, bbar = bbar_n, vbar = vbar_n }


  def rescale' [m] upper_tol lower_tol eps ai ui ki : (M.t, M.t, [m]M.t) =
    if (ai M.>= upper_tol) || (ai M.< lower_tol) then
    let lgai = M.log ai in
    let output = M.((sqrt ai, ui + (eps * lgai / i32 2), map (* (sqrt ai)) ki)) in
	      output 
    else
      (ai, ui, ki)

  def matrix_rescale' upper_tol lower_tol eps a u k =
    let arr = map3 (rescale' upper_tol lower_tol eps) a u k in
    unzip3 arr


  -- Same algorithm, different conventions.
  def normalize_state' [n][m] (st : algo3_state[n][m]) (exp_absorb_cutoff :t) (eps: t):
    algo3_state[n][m]=
    let (abar_n, ubar_n, kbar_n0) =
      -- let _ = if any (M.== zero) st.abar then #[trace] 444 else 0 in
      -- let _ = if any M.isinf st.abar then #[trace] st.abar else st.abar in
      if any (M.>= exp_absorb_cutoff) st.abar ||
	 any (M.<= M.recip exp_absorb_cutoff) st.abar then
	matrix_rescale' exp_absorb_cutoff (M.recip exp_absorb_cutoff) eps st.abar st.ubar st.Kbar
      -- (|> manifest)
      else
      -- let _ = #[trace] count_nan2d st.Kbar in
      (st.abar, st.ubar, st.Kbar)
      -- (|> manifest)
      in
    let (bbar_n, vbar_n, kbar_n) = 
      if any (M.>= exp_absorb_cutoff) st.bbar ||
	 any (M.<= M.recip exp_absorb_cutoff) st.bbar then
      let (bbar2, vbar2, kbar2t) =
	matrix_rescale' exp_absorb_cutoff (M.recip exp_absorb_cutoff) eps st.bbar st.vbar
		       (transpose kbar_n0)
      in
      (bbar2, vbar2, transpose kbar2t)
      else
	(st.bbar, st.vbar, kbar_n0)
    in
    { abar = abar_n, ubar = ubar_n, Kbar = kbar_n, bbar = bbar_n, vbar = vbar_n }

  local def dykstra_helper [n][m] (a : [n][m](M.t)) : ([n][m]M.t, [n]M.t)=
    let avgs = map avg a in
    let a' = map2 sub_vs a avgs in
    (a', avgs)

  -- Takes a matrix x and tries to write it as
  -- x = c + 1p^T + q1^T, where sum_rows c \approx sum_cols c \approx 0.
  -- This is useful because we want to solve optimal transport problems
  -- of the form e^{-C/\varepsilon}, where C may be large and \varepsilon
  -- may be small.
  def dykstra_matrix [n][m] (a : [n][m](M.t)) tol : ([n][m]M.t, [n]M.t, [m]M.t) =
    let (c_final, q_final, p_final0, p_final1) = 
      let (c1, q1) =
  	let (c1_t, q1) = dykstra_helper (transpose a) in
	(transpose c1_t, q1) in
      let (c2, p2) = dykstra_helper c1 in
      -- Loop invariant: c2 + p0 + p2' + q1 = x
      loop (c2 : [n][m]M.t, q1, p0, p2') = (c2, q1, map (\_ -> zero) (0..<n), p2)
      while max_abs p2' M.>= tol do
      let p2 = map2 (M.+) p0 p2' in
      let (c3, q3') =
	let (c3_t, q3') = dykstra_helper (transpose c2) in
	(transpose c3_t, q3') in
      let q3 = map2 (M.+) q1 q3' in
      let (c4, p4') = dykstra_helper c3 in
      (c4, q3, p2, p4') 
    in
    (c_final, map2 (M.+) p_final0 p_final1, q_final)

  -- PROBLEM:
  -- Have a linear cost matrix C \in R^{n\times m}
  -- Have two marginal penalty functions F1 and F2.
  -- F1 and F2 could be based, for example, on Kullback-Leibler divergence,
  -- or they could be infinite for P_X(R) not equal to the specified p,
  -- P_Y(R) not equal to some specified dy.
  -- Want to minimize over R for the entropy-regularized problem -
  -- < C,  R >_F + F1(P_X(R)) + F2(P_Y(R)) = eps * H(R).
  -- SOLUTION:
  -- First, absorb the cost matrix into the entropy term H.
  -- This implements the naive "Algorithm 1" from
  -- the paper.

  def algo1 [n] [m] proxdivF1 proxdivF2
    (dx : [n]t) (dy : [m]t) (K : ([n][m]t)) (eps : t) (tol : t) =
    let proxdivF1' b = proxdivF1 (mat_vec K (odot b dy)) eps in
    let proxdivF2' a = proxdivF2 (mat_vec (transpose K) (odot a dx)) eps in
    let b = replicate m one in
    let (a', b', _) =
      loop (a, b, tolab) = (proxdivF1' b, b, tol) while tolab M.>= tol do
        let a' = proxdivF1' b in
	let b' = proxdivF2' a' in
	let error = (err a a') M.+ (err b b') in
	(a', b', error)
    in
    let K1 = map2 scale K a' in
    let K2 = map2 scale (transpose K1) b' in 
    transpose K2
  -- This algorithm avoids any rescaling, unfortunately, and is so not
  -- numerically stable.

  -- Computes the matrix K_ij = e^{ (u_i + v_j - C_ij) / eps }
  def stabilized_kernel [n] [m] (u: [n]t) (v : [m]t) (C : [n][m]t) eps :[n][m]t =
    (let uplusv = ext_sum u v in
    (let diff = map2 ( map2 (M.-) ) uplusv C in
    (map (map (\x -> M.exp (x M./ eps))) diff)))

  -- This implements "Algorithm 2" from the paper.
  def algo2 [n] [m] proxdivF1 proxdivF2 (dx : [n]t)
    (dy : [m]t) (C : ([n][m]t)) (eps : t) (tol : t)
    (absorption_cutoff : i64) : ([n]t, [m]t, [n]t, [m]t, [n][m]t, t) =
    let proxdivF1' (K_tilde : [n][m]t) (b_tilde : [m]t) (u : [n]t)  =
      proxdivF1 (mat_vec K_tilde (odot b_tilde dy)) u eps in
    let proxdivF2' (K_tilde : [n][m]t) a_tilde v =
      proxdivF2 (mat_vec (transpose K_tilde) (odot a_tilde dx)) v eps in
    let b_tilde = replicate m one in
    let u = replicate n zero in
    let v = replicate m zero in
    let K_tilde = stabilized_kernel u v C eps in    
    let a_tilde = proxdivF1' K_tilde b_tilde u in
    let toobig [k] (u: [k]t) (a:[k]t) : [k]t = map2 (M.+) u (map (M.* eps) a) in
    loop (a, b, u, v, K, error) : ([n]t, [m]t, [n]t, [m]t, [n][m]t, t)=
      (a_tilde, proxdivF2' K_tilde a_tilde v, u, v, K_tilde, tol)
    while error M.>= tol do
    let a' = proxdivF1' K_tilde b u in
    let b' = proxdivF2' K_tilde a' v in
    if
      (let log_compare x = M.log x M.>= M.i64 absorption_cutoff in
      any log_compare a' || any log_compare b')
    then
      let u' = toobig u a in
      let v' = toobig v b in
      let K' = stabilized_kernel u v C eps in
      let b' = replicate m one in
      (a', b', u', v', K', error)
    else
      (a', b', u, v, K, (err b' b) M.+ (err a a'))
  -- The main weakness of this one is that there is no initial stabilization step
  -- which would account for the case where C is large.
  -- Thus, the algorithm simply fails immediately when we give it initial C which is too big.

  -- I will just call this "sinkhorn" as it is
  -- the one we will use most often, I think.
  -- Hm. I will just leave dx and dy in the algorithm.
  -- That may slow things down in some cases.
  -- but it's better than screwing around with it.

   type marginal_distributions [n][m] =
    { dx : [n]M.t,
      dy : [m]M.t }

  -- Notice that a is not supplied to the function.
  -- As an invariant, we assume that b is optimal 
  -- for a and K (a is not supplied here because it is not needed, but b should be optimal
  -- for the version of a from the previous iteration)
  def sinkhorn_update [n][m] (rp: S.reg_params[n][m]) (md : marginal_distributions[n][m])
     (b: [m]M.t) (K : [n][m]t) (u : [n]M.t) (v : [m]M.t)=
     let a1 = S.proxdiv_F1 rp (mat_vec K (map2 (M.*) b md.dy)) u in
     let b1 = S.proxdiv_F2 rp (mat_vec (transpose K) (map2 (M.*) a1 md.dx)) v in
     (a1, b1)

  type stability_params =
    { exp_absorb_cutoff : t, tol_sinkhorn : t, safe_for_exp : t, loop_count : i32 }

  def kbar_of_c_u_v [n][m] (C : [n][m]M.t) (u : [n]M.t) (v: [m]M.t) (p : S.reg_params[n][m]) =
    C
    |> map (map M.neg)
    |> (\x -> map2 add_vs x u)
    |> transpose
    |> (\x -> map2 add_vs x v)
    |> transpose
    |> map (map (M./ (S.eps p)))
    |> map (map M.exp)
    -- map2 (map (map (M./ (M.neg (S.eps p)))) C)

  -- b assumed optimal for some unspecified a.
  -- This should just be run in a loop tightly.
  def sinkhorn_no_check [n][m] (rp : S.reg_params[n][m]) (md : marginal_distributions[n][m])
    (u: [n]M.t) (b0: [m]M.t) (v: [m]M.t)
    (ct : i32)
    (K : [n][m]M.t)
   = loop b = b0 for _i < ct do
     let (_, b') = (sinkhorn_update rp md b K u v)
     in b'

  type^ continue_condition [n] = [n]M.t -> [n]M.t -> i32 -> bool

  def sinkhorn [n][m] (rp : S.reg_params[n][m]) (md : marginal_distributions[n][m])
    (p : stability_params)
    (u0 : [n]M.t) (v0 : [m]M.t)
    C
    (continue : continue_condition[n])
  : ([n]M.t, [m]M.t, [n][m]M.t)

  =
    let (_, _, a_final, u_final, b_final, v_final, K_final) =
      loop (main_loop_count, a_old:[n]M.t, a:[n]M.t, u, b:[m]M.t, v, K) =
	(0: i32, replicate n zero, replicate n one, u0, replicate m one,
	 v0, kbar_of_c_u_v C u0 v0 rp)
      while (continue a_old a main_loop_count) do
      let b' = sinkhorn_no_check rp md u b v p.loop_count K in
      let a' = S.proxdiv_F1 rp (mat_vec K (map2 (M.*) b' md.dy)) u in
      let st = { abar = a', ubar = u, bbar = b', vbar = v, Kbar = K} in
      let { abar = a_new, ubar = u_new, bbar = b_new, vbar = v_new, Kbar = K_new }
      = normalize_state' st p.exp_absorb_cutoff (S.eps rp)
      in
      (main_loop_count + 1, a, a_new, u_new, b_new, v_new, K_new)
    in
    (u_final, v_final, plan_of a_final b_final K_final)
}
