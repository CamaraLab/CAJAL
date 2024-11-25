-- This file implements algorithms from the paper - "The Unbalanced
-- Gromov Wasserstein Distance: Conic Formulation and Relaxation" by
-- Thibault Séjourné, François-Xavier Vialard and Gabriel Peyré

module pairs(M: real) = {
  import "lib/github.com/diku-dk/segmented/segmented"
  -- import "scaling_unbalanced"
    -- This code is a slight modification of replicate iota in the stdlib.
  -- [n-1; n-2; n-3;...;0]
  def pairs (n : i64) =
    let reps = map (\k -> (n - 1) - k) (iota n) in
    let s1 = scan (+) 0 reps in
    let s2 = map (\i -> if i==0 then 0 else s1[i-1]) (iota n) in
    let tmp = scatter (replicate (reduce (+) 0 reps) 0) s2 (iota n) in
    let flags = map (>0) tmp in
    let fst = segmented_scan (+) 0 flags tmp in
    let snd' = segmented_iota flags in
    let snd = map2 (\a b-> a+b+1) fst snd' in
    zip fst snd

  def k_of_i_j n i j =
    let k_of_i n i = i64.(n * i - (i * (i+1))/2) in
    (k_of_i n i) + j - i - 1

  def squareform [z] (a: [z]M.t) (n : i64) : [n][n]M.t =
    tabulate_2d n n (\i j -> if i64.(i<j) then a[k_of_i_j n i j] else if i64.(i>j) then a[k_of_i_j n j i] else M.i32 0)
}

module unbalanced_gw (M : real) = {
  -- import "common"
  -- import "scaling_unbalanced"
  import "unbalanced_gw_core"
  open unbalanced_gw_core M
  -- open common M
  -- module sinkhorn = scaling_unbalanced M
  -- module gw = gromov_wasserstein M
  type t = M.t
  module pairs = pairs(M)
    
  def unbalanced_gw_total_cost [n][m] 
    rho1 rho2 eps (X : [n][n]t) mu (Y : [m][m]t) nu
    p tol_outerloop =
    let (_, _, _, P) =
      unbalanced_gw rho1 rho2 eps X mu Y nu p tol_outerloop
    in
    UGW_cost_arr { rho1, rho2, eps, mu, nu, C = P } X Y

  def unbalanced_gw_pairwise [m][n] (dms: [m][n][n]t) rho1 rho2 eps
    exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop =
    let u = replicate n (M.recip (M.i64 n)) in
    let ugw (i,j) =
      let output = 
      unbalanced_gw_total_cost rho1 rho2 eps dms[i] u dms[j] u
       {exp_absorb_cutoff, loop_count=10, safe_for_exp, tol_sinkhorn} tol_outerloop in
      if M.(output[0] > zero) then output else
      unbalanced_gw_total_cost rho1 rho2 eps dms[trace i] u dms[trace j] u
       {exp_absorb_cutoff, loop_count=1, safe_for_exp, tol_sinkhorn} tol_outerloop	
    in
    map ugw (pairs.pairs m)
    
  def unbalanced_gw_pairwise_pt_clouds [m][n][d] (pt_clouds: [m][n][d]t) =
    let dms = map pdist pt_clouds in unbalanced_gw_pairwise dms

}

module unbalanced_gw64 = unbalanced_gw f64

-- These functions are all related to the original version of the Unbalanced Gromov-Wasserstein
-- algorithm from the paper. Based on our testing we believe that there are no common
-- situations where the algorithm is guaranteed to converge. Experimentally it has worked 
-- for point clouds in low-dimensional Euclidean space but in high-dimensional Euclidean space
-- it seemed to fail for some inputs. For tree geodesics, we have frequently observed
-- failures to converge.

-- These algorithms are not recommended for any purposes. They are faster when they converge
-- but you are warned that you may computing time in an infinite loop.
module original = {
  def unbalanced_gw_init rho1 rho2 eps X mu Y nu init
     exp_absorb_cutoff tol_sinkhorn safe_for_exp tol_outerloop = let (
     _, _, _, P) = unbalanced_gw64.unbalanced_gw_init {rho1, rho2,
     eps, mu, nu, C= init} X Y { exp_absorb_cutoff, tol_sinkhorn,
     loop_count = 10, safe_for_exp } tol_outerloop in P

-- Parameters:
-- rho1, rho2: marginal penalty costs, using notation from
--   the "Unbalanced Gromov-Wasserstein" paper.
--   When rho1 (respectively rho2) are chosen higher, there is more of a cost
--   paid to diverge from the appropriate marginals of the transport plan (create or destroy mass).
--   As rho1, rho2 approach infinity, we converge to usual GW (with regularization parameter epsilon.)
--   As rho1 approaches infinity we converge to a transport plan that embeds the first space
--   itrometrically into the second.
-- epsilon: Entropy penalty, for regularization, using notation from the UGW paper.
-- exp_absorb_cutoff: A numerical stability parameter. The algorithm maintains an internal
--   transport plan whose entries tend to diverge to +\infty or possibly 0.
--   When entries in the matrix for the transport plan exceed exp_absorb_cutoff, or
--   take on a value less than 1/exp_absorb_cutoff, an rescaling step is taken
--   which rescales that row or column to a normal range while absorbing the logarithm
--   of the extreme value into a separately stored array, i.e., the representation (a, x)
--   for a * e^x is replaced with (a * log(m), x-m) for m chosen so that x-m lies in a convenient
--   range. Choose exp_absorb_cutoff to be much less than the maximum value for a floating
--   point number, I have been using 10^30 but probably 10^100 would be okay for 64 bits.
-- safe_for_exp: This is a numerical stability parameter which is used only once at the very
--   beginning of the algorithm to choose a transform of the initial transport plan
--   into safe values. Choose safe_for_exp such that, if -safe_for_exp < x < safe_for_exp,
--   then e^x will be within a reasonable range in (0, \infty). I have been using 30.
-- tol_sinkhorn: This controls the exit condition of the inner Sinkhorn loop, which
--   iteratively modifies the row and column scaling factors of the inner transport plan.
--   The inner loop exits when the arithmetic difference between the old row scaling coefficients
--   and new row scaling coefficients is less than tol_sinkhorn.
-- tol_outerloop: This controls the exit condition of the outer loop, which is minimizing the
--   overall unbalanced Gromov-Wasserstein cost. It controls the arithmetic error between
--   successive transport plans (directly, not in the form of the scaling coefficients.)
  def unbalanced_gw_total_cost rho1 rho2 eps X mu Y nu
    exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop
  =
  unbalanced_gw64.unbalanced_gw_total_cost rho1 rho2 eps X mu Y nu
  { exp_absorb_cutoff, safe_for_exp, loop_count = 10, tol_sinkhorn} tol_outerloop

  def unbalanced_gw_pairwise = unbalanced_gw64.unbalanced_gw_pairwise

  def unbalanced_gw_pairwise_pt_clouds = unbalanced_gw64.unbalanced_gw_pairwise_pt_clouds
  
  def ugw_cost_arr rho1 rho2 eps X mu Y nu C =
    unbalanced_gw64.UGW_cost_arr {rho1, rho2, eps, mu, nu, C} X Y

  def init_step rho1 rho2 eps X mu Y nu init
     exp_absorb_cutoff tol_sinkhorn safe_for_exp
   =
    let ( _, _, P) = 
      unbalanced_gw64.unbalanced_gw_init_step
      {rho1, rho2, eps, mu, nu, C= init} X Y
      { exp_absorb_cutoff, tol_sinkhorn, loop_count = 10, safe_for_exp }
  in P

  def init_step0 rho1 rho2 eps X mu Y nu init exp_absorb_cutoff
     tol_sinkhorn safe_for_exp
= let ( _, _, P) =
    unbalanced_gw64.unbalanced_gw_init_step
    {rho1, rho2, eps, mu, nu, C= init} X Y
    { exp_absorb_cutoff, tol_sinkhorn, loop_count = 10, safe_for_exp }
  in
  unbalanced_gw64.UGW_cost_arr {rho1, rho2, eps, mu, nu, C= P} X
}

-- These functions are related to "naive" gradient descent.
-- They should be correct. In our experience they perform poorly compared to the others
-- and are not worth using. They are included only for benchmarking.
module naive = {
  module pairs = pairs(f64)
  def ugw_naive rho1 rho2 eps A mu B nu tau c tol_outerloop =
    unbalanced_gw64.gradient_descent.naive_descent rho1 rho2 eps A mu B nu {tau, c} tol_outerloop

  def ugw_naive_pairwise [k][m] (A: [k][m][m]f64.t) rho1 rho2 eps tau c tol_outerloop =
   let mu = replicate m (1.0f64 / (f64.i64 m))  in
    pairs.pairs k
   |>  map(\(i,j) -> unbalanced_gw64.gradient_descent.naive_descent rho1 rho2 eps A[i] mu A[j] mu {tau, c} tol_outerloop)
}

module pairsf64= pairs(f64)
entry ugw_armijo_pairwise [k][m] rho1 rho2 eps (A: [k][m][m]f64.t) (distrs: [k][m]f64.t) exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop =
  pairsf64.pairs k
  |> map (\(i,j) ->
	unbalanced_gw64.armijo.main
	rho1 rho2 eps A[i] distrs[i] A[j] distrs[j]
       {exp_absorb_cutoff, loop_count=1, safe_for_exp, tol_sinkhorn} tol_outerloop)

entry ugw_armijo_pairwise_increasing [k][m] (ugw_dmat:[k][k]f64)(ratio:f64)
  rho1 rho2 eps (A: [k][m][m]f64.t) (distrs: [k][m]f64.t) exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop =
  pairsf64.pairs k
  |> map (\(i,j) ->
	    loop (current_ugw, current_epsilon) = (ugw_dmat[i][j], eps) while f64.isnan current_ugw do
	    let increase_eps = current_epsilon f64.* ratio in
	    let arr = unbalanced_gw64.armijo.main 
		      rho1 rho2 increase_eps A[i] distrs[i] A[j] distrs[j]
		      {exp_absorb_cutoff, loop_count=1, safe_for_exp, tol_sinkhorn} tol_outerloop
	    in
	    (f64.(arr[0] + (arr[1] * rho1) + (arr[2]) * rho2), increase_eps)
	 ) |> unzip |> (\o -> o.0) |> (\a -> pairsf64.squareform a k)

entry ugw_armijo rho1 rho2 eps A mu B nu exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop =
  ugw_armijo_pairwise rho1 rho2 eps [A,B] [mu, nu]
		      exp_absorb_cutoff safe_for_exp tol_sinkhorn tol_outerloop
		      |> (\x -> x[0])

entry ugw_armijo_pairwise_unif [k][m] rho1 rho2 eps (A: [k][m][m]f64.t) =
  let mu = replicate m (1.0f64 / (f64.i64 m)) in
  let distrs = replicate k mu in 
  ugw_armijo_pairwise rho1 rho2 eps A distrs 

entry ugw_armijo_euclidean [k][m][d] rho1 rho2 eps (pt_clouds: [k][m][d]f64.t) =
  let dmats = map unbalanced_gw64.pdist pt_clouds in 
  ugw_armijo_pairwise_unif rho1 rho2 eps dmats
