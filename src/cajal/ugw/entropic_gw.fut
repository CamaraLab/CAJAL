-- This file implements the entropic Gromov-Wasserstein function.

module entropic_gw(M : real) = {
  import "common"
  import "scaling_unbalanced"
  import "linear_ot"
  import "gw"

  module su = scaling_unbalanced (M)
  module gw = gromov_wasserstein (M)

  open common M
  open linear_ot M

  type params = { tol_inner : M.t, tol_outer : M.t, eps : M.t, safe_for_exp_thresh : M.t }

  def cost_matrix_of X Y P =
    c.mat_mul (c.mat_mul (map (map (M.* (M.neg (M.i32 2)))) X) P) Y

  def entropic_gw_update [n][m] (n2X : [n][n]M.t) (mu :[n]M.t) (Y : [m][m]M.t) (nu :[m]M.t)
           (P : [n][m]M.t) (p : params) =
    if i32.sum ( (map su.count_nan P)) == 0 then
    let C = su.mat_mul (su.mat_mul n2X P) Y in
    sinkhorn C mu nu
	     { eps = p.eps,
	       safe_for_exp_thresh = p.safe_for_exp_thresh,
	       tol = p.tol_inner,
	       ct = 10 }
    else P
  def n2 = map (map M.((* (neg (i32 2)))))

  def entropic_gw [n][m] (X : [n][n]M.t) (mu :[n]M.t) (Y : [m][m]M.t) (nu :[m]M.t)
    (init : [n][m]M.t) (p :params)=
    let n2X = n2 X in
    (loop (P0, P1) =
       (init, entropic_gw_update (n2X) mu Y nu init p)
     while any (\x -> x M.> p.tol_outer) (map2 ratio_err P0 P1)
     do
       -- let _ = #[trace] "!!!!!!!!!!!!!!!!!!!!" in
       (P1, entropic_gw_update (n2X) mu Y nu P1 p))
    |> (\(_, a) -> a)

  def entropic_gw_cost [n][m] (X : [n][n]M.t) (Y : [m][m]M.t)
    (P : [n][m]t)
    (eps : M.t) =
    gw.GW_cost X Y P M.+ (eps M.* H P)

}
