open Multicore.Unbalanced_gw_multicore
open Bigarray

module Context = Context

module Array_f64_1d = Array_f64_1d
module Array_f64_2d = Array_f64_2d

val uniform: int -> (float, float64_elt, c_layout) Genarray.t


(** For documentation on the meaning of the parameters, see the README.org,
    or the documentation for the unbalanced_gw_total_cost function in unbalanced_gw.fut. *)
type params = { rho1 : float; rho2 : float; epsilon : float;
               exp_absorb_cutoff : float;
               safe_for_exp : float;
               tol_sinkhorn : float;
               tol_outerloop : float
             }

module Vectorform : sig
  type t = private (float, float64_elt, c_layout) Genarray.t
  val num_pts_t: t -> int
  type arr
  val num_pts_arr: arr -> int
  val arr_of_npy : string -> arr
  val arr_of_npz : string -> string -> arr
  val to_file : t -> string -> unit
end

module Squareform: sig
  type t
  type arr
  val num_pts_t: t -> int
  val num_pts_arr: arr -> int
  val of_vectorform : Vectorform.t -> t
  val of_vectorform_arr : Vectorform.arr -> arr    
  val num_spaces: arr -> int
  val t_of_npy: string -> int option -> t
  val arr_of_npy: string -> int option -> arr
  val unbalanced_gw_armijo : 
     Context.t -> 
     t -> Array_f64_1d.t -> 
     t -> Array_f64_1d.t ->
     params -> 
    (float, float64_elt, c_layout) Genarray.t   
  val unbalanced_gw_armijo_pairwise_unif : Context.t -> 
    arr -> params -> (float, float64_elt, c_layout) Genarray.t
  
  val slice_left: arr -> int -> t
  val sub_left: arr -> int -> int -> arr
  val ugw_armijo_pairwise_increasing :
    Context.t ->
    original_ugw_vform_dmat:(float, float64_elt, c_layout) Genarray.t ->
    increasing_ratio:float ->
    arr ->
    params ->
    (float, float64_elt, c_layout) Genarray.t
end

module Pt_cloud : sig
  type t
  type arr
  val num_pts_t: t -> int
  val num_spaces: arr -> int
  val t_of_npy: string -> t
  val arr_of_npy: string -> int option -> arr
  val arr_of_npy_dir : string -> int option -> arr
  val num_pts_arr: arr -> int
  val slice_left: arr -> int -> t
  val sub_left: arr -> int -> int -> arr
  val pdist : t -> Squareform.t
  val map_pdist : arr -> Squareform.arr
end
