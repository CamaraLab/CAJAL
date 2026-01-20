-- This file implements algorithms from the paper "Scaling algorithms
-- for unbalanced optimal transport"
-- by CHIZAT, GABRIEL PEYR´E, BERNHARD SCHMITZER,
-- AND FRANCOIS-XAVIER VIALARD
-- and the paper
-- "The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation"
-- by Thibault Séjourné, François-Xavier Vialard and
-- Gabriel Peyré
-- From page 22.

module scaling_unbalanced (M : real) = {
  import "common"
  import "scaling_generic"

  type coeff = { rho1 : M.t, rho2 : M.t, eps : M.t }
  type otp [n][m] = { rho1 : M.t, rho2 : M.t, eps : M.t, mu : [n]M.t, nu : [m]M.t, C :[n][m]M.t }

  module su_problem = {
    type t = M.t
    type reg_params [n][m] = otp[n][m]
    def eps [n][m] (a : reg_params[n][m]) : t = a.eps

    -- Returns abar (a = abar * e^ubar)
    open common M

    def proxdiv_F1 [n][m] (rp : reg_params[n][m]) (kb : [n]t) (ubar: [n]M.t) : [n]t =
      let c1 = (M.neg rp.rho1) M./ (rp.rho1 M.+ rp.eps) in
      let c2 = (M.neg rp.eps) M./ (rp.rho1 M.+ rp.eps) in
      let kb_mu_pow = map2 (M./) kb rp.mu |> map (M.** c1) in
      let e_ubar_scale = map (M.* c2) ubar |> (map M.exp) in
      map2 (M.*) kb_mu_pow e_ubar_scale

    def proxdiv_F2 [n][m] (rp : reg_params[n][m]) (kt_a : [m]t) (vbar: [m]M.t) : [m]t =
      let c1 = (M.neg rp.rho2) M./ (rp.rho2 M.+ rp.eps) in
      let c2 = (M.neg rp.eps) M./ (rp.rho2 M.+ rp.eps) in
      let kt_a_nu_pow = map2 (M./) kt_a rp.nu |> map (M.** c1) in
      let e_vbar_scale = map (M.* c2) vbar |> (map M.exp) in
      map2 (M.*) kt_a_nu_pow e_vbar_scale
  }

  open scaling_generic M su_problem
  -- type otp [n][m] = { rho1 : t, rho2 : t, eps : t, mu : [n]t, nu : [m]t, C :[n][m]t }

  -- Entropic cost. Defined as
  -- < C, P > + rho1 * KL(pi_1 P | mu) + rho2 * KL(pi_2 P | nu) + eps H(P)
  -- Double check your conventions about whether
  -- you need to use H(P) or KL(P | mu \otimes nu).
  -- This is subtle, minimizing H(P) biases us toward the uniform
  -- distribution, whereas
  -- minimizing KL(P | mu \otimes nu) biases us towards mu \otimes nu.

  def cost [n][m] (r : otp [n][m]) (P : [n][m]t) = M.(
    (sum (map2 dot r.C P)) +
    ( r.eps * (H P)) +
    r.rho1 * (KL (map sum P) r.mu) +
    r.rho2 * (KL (map sum (transpose P)) r.nu)
  )

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

  -- Returns abar (a = abar * e^ubar)
  def proxdiv3 [n] [m] (kbar : [n][m]M.t) (bbar : [m]M.t) (ubar : [n]M.t) (mu : [n]M.t)
     (rho1 : M.t) (eps : M.t) =
    let c1 = (M.neg rho1) M./ (rho1 M.+ eps) in
    let c2 = (M.neg eps) M./ (rho1 M.+ eps) in
    let kb = mat_vec kbar bbar in
    let kb_mu_pow = map2 (M./) kb mu |> map (M.** c1) in
    let e_ubar_scale = map (M.* c2) ubar |> (map M.exp) in
    map2 (M.*) kb_mu_pow e_ubar_scale

  def ct_zeros a = reduce (i32.+) 0
	     (map (\x -> if x M.==  (M.i32 0) then 1 else 0) a)

  def algo3_update [n][m] (st : algo3_state[n][m]) (r : otp[n][m])
    (exp_absorb_cutoff :t) : algo3_state [n][m]=
    let bbar' = proxdiv3 (transpose st.Kbar) st.abar
			 st.vbar r.nu r.rho2 r.eps in
    let abar' = proxdiv3 st.Kbar bbar' st.ubar r.mu r.rho1 r.eps in
    normalize_state { abar = abar', vbar = st.vbar, bbar = bbar',
      ubar = st.ubar, Kbar = st.Kbar } exp_absorb_cutoff

  def algo3_core_loop [n][m] (st : algo3_state[n][m]) (r : otp[n][m])
    (params : stability_params) =
    let (st1 : algo3_state[n][m]) = algo3_update st r params.exp_absorb_cutoff in
    loop (st0 : algo3_state[n][m], st1 : algo3_state[n][m]) = (st, st1)
    while ((ratio_err st0.abar ( st1.abar))) M.>= params.tol_sinkhorn do
    let st2 = algo3_update st1 r params.exp_absorb_cutoff in
    (st1, st2)

  def simple_update [n][m] (st : algo3_state[n][m]) (r : otp[n][m]): algo3_state [n][m]=
    let bbar' = proxdiv3 (transpose st.Kbar) st.abar
			 st.vbar r.nu r.rho2 r.eps in
    let abar' = proxdiv3 st.Kbar bbar' st.ubar r.mu r.rho1 r.eps in
    {Kbar = st.Kbar, abar = abar', bbar = bbar', ubar = st.ubar, vbar = st.vbar }

  def ratio_err_ok tol a b = 
    M.(a * (one + tol) >= b && b * (one + tol) >= a)

  def any2 f a b =
      reduce_comm (||) false (map2 f a b)
    
  -- Some performance tweaks.
  def algo5_core_loop [n][m] (st : algo3_state[n][m]) (r : otp[n][m])
    (params : stability_params) =
    let (st1 : algo3_state[n][m]) =
      loop (st : algo3_state[n][m]) = st for _i < params.loop_count do
	algo3_update st r params.exp_absorb_cutoff
    in
    let no_nans_abar state = (all (\x -> not (M.isnan x)) state.abar) in
    let still_converging st0 st1 = (any2 (\x y -> not (ratio_err_ok params.tol_sinkhorn x y)) st1.abar st0.abar)

    let (st0, st1) = (
	loop (st0 : algo3_state[n][m], st1 : algo3_state[n][m]) = (st, st1)
	while no_nans_abar st1 && still_converging st0 st1 do
	let st_inner : algo3_state[n][m] =
	  (loop st_inner = st1 for _i < params.loop_count do simple_update st_inner r)
	in (st1, normalize_state st_inner params.exp_absorb_cutoff)
      ) in
  if no_nans_abar st1 then (st0, st1) else
    let (st0, st1) = (
	loop (st0 : algo3_state[n][m], st1 : algo3_state[n][m]) = (st0, algo3_update st0 r params.exp_absorb_cutoff)
	while no_nans_abar st1 && still_converging st0 st1 do
	let st_inner : algo3_state[n][m] =
	  (loop st_inner = st1 for _i < params.loop_count do algo3_update st_inner r params.exp_absorb_cutoff)
	in (st1, st_inner)
      ) in
    (st0, st1)
    
  -- This algorithm is similar to algo2.
  -- It takes the same list of parameters.
  -- However, instead of calculating K directly, it first
  -- "normalizes" C by offloading some of the mass into row and
  -- column vectors. Algorithm 2 stops a and b from becoming extreme,
  -- but it does not account for the possibility that K may
  -- already be so small/large that one has immediate underflow/overflow.
  -- For readability I will hardcode this one to the
  -- unbalanced linear optimal transport problem.
  -- I can always refactor it later for greater generality.
  -- Another difference is that I have changed 
  -- from writing a = \tilde{a} * e^{\tilde{u}/\varepsilon} to a = \bar{a} * e^{\bar{u}}
  -- dropping varepsilon from the definition.

  def algo3 [n][m] (r : otp[n][m]) (params : stability_params) =
  let c0 = map (map (M./ (M.neg r.eps))) r.C in
    -- let (cbar, u0, v0) =
    -- let (cbar, u0_neg, v0_neg) = dykstra_matrix c0 params.safe_for_exp in
    -- (cbar, map (M.neg) u0_neg, map M.neg v0_neg)
  let (cbar, u0, v0) = safe_for_exp c0 params.safe_for_exp in
  let kbar = map (map (M.exp)) cbar in
  let bbar = map (\_ -> one) (0..<m) in
  let (abar : [n]M.t) = proxdiv3 kbar bbar u0 r.mu r.rho1 r.eps in
  let (_,s) = algo3_core_loop
	  { abar = abar, ubar = u0, bbar = bbar, vbar = v0, Kbar = kbar }
	  r params
  in
  (s.ubar, s.vbar, map2 scale s.Kbar s.abar |> transpose |>
		   (\k -> map2 scale k s.bbar) |> transpose)

  -- Returns an initial state for the Sinkhorn loop,
  -- including a normalization of the input matrix that reduces it to a safe range.
  def init_state [n][m] (r : otp[n][m]) (abar0 : [n]t) (ubar0 : [n]t)
            (bbar0 : [m]t) (vbar0 : [m]t) : algo3_state[n][m] =
    let cbar =
      let c0 = map (map (M./ (M.neg r.eps))) r.C in
      let c1 = map2 (\a v -> map (a M.+) v) ubar0 c0 |> transpose in
      map2 (\a v -> map (a M.+) v) vbar0 c1 |> transpose
    in
    -- let kbar = map (map M.exp) cbar in
    -- M.({abar = abar0, ubar = ubar0, bbar = bbar0, vbar = vbar0, Kbar = kbar})
    let (cbar', u', v') = safe_for_exp cbar (M.i64 30) in
    let kbar = map (map M.exp) cbar' in
    M.({abar = abar0, ubar = map2 (+) ubar0 u', bbar = bbar0,
     vbar = map2 (+) vbar0 v', Kbar = kbar })

  -----------------------------------------------
  --- algo4, final version of the algorithm.
  -- This improves upon the previous versions of the algorithm by considering
  -- that we may already have an estimate of the transport plan, i.e.,
  -- what are the appropriate (abar, ubar, bbar, vbar, Kbar).
  -- We do not need the original K if we have ubar, vbar and Kbar.
  -- One can always reconstruct K = Kbar/ubar * vbar^T.
  --
  -- OUTLINE:
  -- The user wants to solve the problem of finding the optimal coupling matrix
  -- P (an nxm matrix) between two marginal distributions mu, nu, given
  -- an nxm cost matrix C, minimizing the cost function
  -- Cost(C) = <C,P> + rho1 * KL(pi_X P | mu ) + rho2 * KL(pi_Y P | nu) + epsilon * H(P)
  -- where H(P) and KL are defined in common.fut.
  -- The problem is simplified by seeking an answer of the form
  -- P = AKB where A, B are square diagonal matrices (n x n and m x m resp.)
  -- and K = e^{-C/epsilon} where epsilon is a small positive number.
  -- This reduces the number of parameters in the search space to m + n
  -- rather than mn.
  -- We write 'a' for the diagonal vector of A, of length n.
  -- We write 'b' for the diagonal vector of B, of length m.
  -- As epsilon grows small, the entries of K will rapidly approach zero,
  -- and the solutions a and b will be correspondingly large.
  -- Therefore we store a "normalized" coefficient a_bar (which is within a few orders of magnitude of 1)
  -- and an exponent u_bar, with a = a_bar * e^u_bar.
  -- Similarly b_bar and v_bar, b = b_bar * e^v_bar.
  -- The user supplies their best guess for a_bar, u_bar, b_bar and v_bar,
  -- which is useful in the case where the user has to solve a series of
  -- similar problems (where the answer to the previous one is close to the answer to the current one)
  -- because you can use the previous answer to speed up the computation.
  -- The algorithm successively computes values of (a_bar, u_bar) and (b_bar, v_bar),
  -- stopping when the previous pair (a_bar, u_bar) is close to the current pair (a_bar', u_bar')
  -- and in this case it returns (a_bar', u_bar') together with the previous (b_bar, v_bar) for that
  -- pair. The returned values for a_bar', u_bar' are guaranteed to minimize the component of the penalty
  -- which is associated to a_bar' and u_bar'.

  def algo4 [n][m] (r : otp[n][m]) (abar0 : [n]t) (ubar0 : [n]t)
            (bbar0 : [m]t) (vbar0 : [m]t) (params: stability_params) =
    let s = init_state r abar0 ubar0 bbar0 vbar0 in
    -- let cbar =
    --   let c0 = map (map (M./ (M.neg r.eps))) r.C in
    --   let c1 = map2 (\a v -> map (a M.+) v) ubar0 c0 |> transpose in
    --   map2 (\a v -> map (a M.+) v) vbar0 c1 |> transpose
    -- in
    -- let kbar = map (map M.exp) cbar in
    let (_, st) = algo5_core_loop s (r : otp[n][m]) params
    in
    -- let _ = #[trace] count_nan st.abar in
    (map2 (M.+) st.ubar (map M.log st.abar),
     map2 (M.+) st.vbar (map M.log st.bbar),
     map2 scale st.Kbar st.abar |> transpose |>
     (\k -> map2 scale k st.bbar) |> transpose)

  def prox_div_strict_s p _ _ = p
  def prox_div_KLrho_s lambda p s eps =
    let c1 = eps M./ (lambda M.+ eps) in
    let c2 = lambda M./ (lambda M.+ eps) in
    odot
    (map (\x -> x M.** c1) s)
    (map (\x -> x M.** c2) p)

  def prox_div_strict_su p s _ _ = map2 (M./) p s
  def prox_div_KLrho_su lambda p s u eps =
    let c1 = lambda M./(lambda M.+ eps) in
    let c2 = M.neg (lambda M.+ eps) in
    odot
    (map (\x -> x M.** c1) (map2 (M./) p s))
    (map M.exp (map (\x -> x M./ c2) u))
}
