-- This file implements functionality which is in general
-- necessary for Gromov-Wasserstein.

module gromov_wasserstein (M : real) ={
  import "common"

  type t = M.t
  open common M	   

  -- ll = local linearization
  -- If T is a coupling matrix, A and B are distance matrices,
  -- a and b are the distributions, then
  -- this function computes L(A,B)\otimes T
  -- such that <L(A,B) \otimes T, T>_F is the classical GW cost of T
  -- (ignoring that T need not be a coupling, properly)
  -- Actually L2_otimes_T does not invoke the distributions.
  -- In the classical case where we want the marginals to agree
  -- with the given distribution on the nose, the computation can be
  -- simplified somewhat.

  def L2_otimes_T [n][m] (A : [n][n]t) (B :[m][m]t) (P :[n][m]t) =
    let a = map M.sum P in
    let b = map M.sum (transpose P) in
    let X2x X x =
      map (\a -> dot (odot a a) x) X in
    let A2a = X2x A a in
    let B2b = X2x B b in
    let n2APB = mat_mul (mat_mul A P) B |> map (map (M.* (M.neg (M.i32 2)))) in
    let A2a_n2APB = map2 add_vs n2APB A2a in
    let A2a_B2b_n2APB = map2 add_vs (transpose A2a_n2APB) B2b |> transpose in
    A2a_B2b_n2APB

  -- Classical GW of the transport plan
  def GW_cost A B P =
    let a = map M.sum P in
    let b = map M.sum (transpose P) in
    let X2xx X x =
      map (\s -> dot (odot s s) x) X |> dot x in
    let A2aa = X2xx A a in
    let B2bb = X2xx B b in
    mat_mul (mat_mul A P) B |> frobenius P |>
    (\s -> (M.i64 (-2)) M.* s M.+ A2aa M.+ B2bb)

  -- The gradient of GW_cost(P), such that
  -- GW_cost(P + dP) \approx GW_cost(P) + frobenius P (nabla_G A B P) for small dP.
  def nabla_G A B P = map (map (M.* M.i32 2)) (L2_otimes_T A B P)

  -- Quadratic GW of two transport plans
  def GW_cost' A B P Q =
    let ap = map M.sum P in
    let bp = map M.sum (transpose P) in
    let aq = map M.sum Q in
    let bq = map M.sum (transpose Q) in
    let X2xy X x y =
      map (\s -> dot (odot s s) x) X |> dot y 
    in
    let A2aa' = X2xy A ap aq in
    let B2bb' = X2xy B bp bq in
        mat_mul (mat_mul A P) B |> frobenius Q |>
    (\s -> (M.i64 (-2)) M.* s M.+ A2aa' M.+ B2bb')
    
}
