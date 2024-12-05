open Bigarray
module Unbalanced_gw = Multicore.Unbalanced_gw_multicore
module Context = Unbalanced_gw.Context

module Array_f64_1d = Unbalanced_gw.Array_f64_1d
module Array_f64_2d = Unbalanced_gw.Array_f64_2d
module Array_f64_3d = Unbalanced_gw.Array_f64_3d

(* type ugw_cost_data = {
  gw_cost: float;
  marginal_1: float;
  marginal_2: float;
  kl_div: float;
  total_cost: float
} *)

let bigarray_of_npy filename =
  Npy.read_mmap2 filename ~shared:false
  |> Npy.to_bigarray2 c_layout Float64
  |> function
    | None -> raise (Invalid_argument "Help")
    | Some bigarray ->
       genarray_of_array2 bigarray
    ;;
    
let bigarray_of_npz filename entry =
  Npy.Npz.read (Npy.Npz.open_in filename) entry
  |> Npy.to_bigarray c_layout Float64
  |> function
    | None -> raise (Invalid_argument "The npz file could not be read.")
    | Some bigarray -> bigarray

let read_dir_i (readfile : int -> string -> 'a) (dir : string) (n : int option)=
  let filenames =
    match n with
    | None -> Sys.readdir dir
    | Some k -> Array.sub (Sys.readdir dir) 0 k
  in
  let pipeline i a = a |> Filename.concat dir |> readfile i in
  Seq.mapi pipeline (Array.to_seq filenames)

let copy_to index filename bigarray () =
  Genarray.blit
  (bigarray_of_npy filename)
  (Genarray.slice_left bigarray [| index |] )

let uniform k = Genarray.init Float64 c_layout [| k |]
    (fun _ -> 1.0 /. (float_of_int k ))

type params = { rho1 : float; rho2 : float; epsilon : float;
               exp_absorb_cutoff : float;
               safe_for_exp : float;
               tol_sinkhorn : float;
               tol_outerloop : float
             }

module Vectorform : sig
  type t = (float, float64_elt, c_layout) Genarray.t
  val num_pts_t : t -> int 
  type arr
  val num_pts_arr : arr -> int 
  val slice_left : arr -> int -> t
  val n_pt_clouds : arr -> int
  val get : t -> int -> int -> int -> float
  val arr_of_npy : string -> arr
  val arr_of_npz : string -> string -> arr
  val to_file : t -> string -> unit
end = struct
  open Bigarray
  type t = (float, float64_elt, c_layout) Genarray.t
  type arr = (float, float64_elt, c_layout) Genarray.t

  let n_pts_k m = 
    let n = Int.of_float @@ Float.ceil @@ Float.sqrt @@ float_of_int @@ 2 * m in
    assert (n * (n-1) = 2 * m); n

  let num_pts_t a = 
    let m = (Bigarray.Genarray.dims a).(0) in
    n_pts_k m
  ;;
let num_pts_arr a = 
    let m = (Bigarray.Genarray.dims a).(1) in
    n_pts_k m
;;
  let slice_left arr k = Bigarray.Genarray.slice_left arr [|k|];;
  let n_pt_clouds : arr -> int = fun arr -> (Bigarray.Genarray.dims arr).(0);;
  let arr_of_npy filename : arr =
  let result = bigarray_of_npy filename in
  let m = Genarray.nth_dim result 1 in
  let n = Int.of_float @@ Float.ceil @@ Float.sqrt @@ float_of_int @@ 2 * m in
    assert (n * (n-1) = 2 * m);
    result
  let to_file : t -> string -> unit = fun arr str -> 
    Npy.write arr str

  let arr_of_npz filename dict_name : arr =
  let result = bigarray_of_npz filename dict_name in
  let m = Genarray.nth_dim result 1 in
  let n = Int.of_float @@ Float.ceil @@ Float.sqrt @@ float_of_int @@ 2 * m in
    assert (n * (n-1) = 2 * m);
    result
  
  let k_of_i ~n ~i = n * i - (i * (i+1))/2;;

  let coords ~n ~i ~j =
    if i < j then 
      (k_of_i ~n ~i) + j - i - 1 else
    if j < i then 
      (k_of_i ~n ~i:j) + i - j - 1 else
        failwith "i=j"
      
  let get : t -> int -> int -> int -> float = 
    fun arr n i j ->
      if i = j then 0. else Bigarray.Genarray.get arr [|coords ~n ~i ~j|]

end

module Squareform : sig
  type t = private (float, Bigarray.float64_elt, c_layout) Bigarray.Genarray.t

  val of_vectorform: Vectorform.t -> t
  type arr
  val of_vectorform_arr: Vectorform.arr -> arr
  val t_of_npy: string -> int option -> t
  val arr_of_npy: string -> int option -> arr
  val num_pts_t: t -> int
  val num_pts_arr: arr -> int   
  val num_spaces : arr -> int
    
  val unbalanced_gw_armijo : 
     Context.t -> 
     t -> Array_f64_1d.t -> 
     t -> Array_f64_1d.t ->
     params -> 
     (float, float64_elt, c_layout) Bigarray.Genarray.t

  val t_of_bigarray: (float, float64_elt, c_layout) Genarray.t -> t
  
  val arr_of_bigarray: (float, float64_elt, c_layout) Genarray.t -> arr

  val slice_left: arr -> int -> t

  (** sub_left arr offset length has initial dimension length *)
  val sub_left : arr -> int -> int -> arr

  val unbalanced_gw_armijo_pairwise_unif : 
  Context.t -> 
  arr ->
  params -> 
  (float, float64_elt, c_layout) Genarray.t

  val ugw_armijo_pairwise_increasing :
    Context.t ->
    original_ugw_vform_dmat:(float, float64_elt, c_layout) Genarray.t ->
    increasing_ratio:float ->
    arr ->
    params ->
    (float, float64_elt, c_layout) Genarray.t
end = struct
  open Bigarray
  type t = (float, float64_elt, c_layout) Genarray.t

  type arr = (float, float64_elt, c_layout) Genarray.t
  let t_of_npy str n =
    (Npy.read_copy str) |> Npy.to_bigarray c_layout Float64 |>
    Option.get
    |> fun a -> let dims = (Genarray.dims a) in
    if (dims.(0) = dims.(1)) then
        match n with 
        | Some l -> Genarray.sub_left a 0 l
        | None -> a
          else failwith "Expected a squareform distance matrix."
  ;;
    
  let arr_of_npy str n = 
    (Npy.read_copy str) |> Npy.to_bigarray c_layout Float64 |>
    Option.get
    |> fun a -> let dims = (Genarray.dims a) in
        assert(dims.(1) = dims.(2));
        match n with 
        | Some l -> Genarray.sub_left a 0 l
        | None -> a
  ;;
  let of_vectorform a = 
    let m = Genarray.nth_dim a 1 in
    let n = Int.of_float @@ Float.ceil @@ Float.sqrt @@ float_of_int @@ 2 * m in
    Bigarray.Genarray.init Float64 c_layout [|n; n|]
    (let n = Vectorform.num_pts_t a in fun arr -> Vectorform.get a n arr.(0) arr.(1))
  
  let of_vectorform_arr a = 
    let m = Vectorform.n_pt_clouds a in 
    let n = Vectorform.num_pts_arr a in 
    Bigarray.Genarray.init Float64 c_layout [|m;n;n|] 
    (fun coords -> let open Vectorform in get (slice_left a (coords.(0))) n (coords.(1)) (coords.(2)))
  ;;

  let num_pts_t t = Genarray.nth_dim t 0
  let num_pts_arr arr = Genarray.nth_dim arr 1
  let num_spaces arr = Genarray.nth_dim arr 0

  let unbalanced_gw_armijo ctx x mu y nu params =
    let open Unbalanced_gw in
    let a_dmat = Array_f64_2d.v ctx x in
    let b_dmat = Array_f64_2d.v ctx y in
    Unbalanced_gw.ugw_armijo
        ctx params.rho1 params.rho2 params.epsilon a_dmat mu b_dmat nu
        params.exp_absorb_cutoff
        params.safe_for_exp
        params.tol_sinkhorn
        params.tol_outerloop
  |> Array_f64_1d.get
;;

let unbalanced_gw_armijo_pairwise_unif ctx arr params =
    let dmats = Array_f64_3d.v ctx arr in
    Unbalanced_gw.ugw_armijo_pairwise_unif
        ctx params.rho1 params.rho2 params.epsilon dmats
        params.exp_absorb_cutoff
        params.safe_for_exp
        params.tol_sinkhorn
        params.tol_outerloop
  |> Array_f64_2d.get

let ugw_armijo_pairwise_increasing ctx
    ~original_ugw_vform_dmat
    ~(increasing_ratio:float)
    arr
    params =
  let original_ugw_dmat = Array_f64_2d.v ctx original_ugw_vform_dmat in
  let dmats = Array_f64_3d.v ctx arr in
  let u = Bigarray.Genarray.(init Float64 c_layout [|nth_dim arr 0; nth_dim arr 1|]
                               (fun _ -> 1./.(Float.of_int (nth_dim arr 1)) )) |> Array_f64_2d.v ctx
  in
    Unbalanced_gw.ugw_armijo_pairwise_increasing 
      ctx original_ugw_dmat increasing_ratio params.rho1 params.rho2 params.epsilon dmats u
        params.exp_absorb_cutoff
        params.safe_for_exp
        params.tol_sinkhorn
        params.tol_outerloop
  |> Array_f64_2d.get

  let t_of_bigarray: (float, float64_elt, c_layout) Genarray.t -> t = fun a ->
    assert (Genarray.nth_dim a 0 = Genarray.nth_dim a 1);
    a
  (* let bigarray_of_t = fun a -> a *)
  
  let arr_of_bigarray: (float, float64_elt, c_layout) Genarray.t -> t = fun a ->
      assert (Genarray.dims a |> Array.length = 3);
      assert (Genarray.nth_dim a 1 = Genarray.nth_dim a 2);
      a

  let slice_left arr i = Genarray.slice_left arr [| i |];;
  let sub_left arr offset length = Genarray.sub_left arr offset length;;
end


module Pt_cloud : sig
  type t
  type arr
  val num_pts_t: t -> int    
  val num_spaces: arr -> int
  val num_pts_arr: arr -> int 
  val arr_of_npy_dir : string -> int option -> arr
  val t_of_npy: string -> t
  val arr_of_npy: string -> int option -> arr
  
  val slice_left: arr -> int -> t

  (** sub_left arr offset length has initial dimension length *)
  val sub_left: arr -> int -> int -> arr
    
  val pdist : t -> Squareform.t
  val map_pdist : arr -> Squareform.arr
end = struct
  open Bigarray

  (** A point cloud is a two dimensional Bigarray of shape (n,d) where
      n is the number of points, and d is the dimension of the
      underlying space. *)
  type t = (float, float64_elt, c_layout) Genarray.t

  (** A point cloud array is a three dimensional Bigarray of shape
      (m, n, d). m is the number of point clouds, n is the number of
      points in each space, and d is the dimension of the Euclidean
      space from which the points are drawn. *)
  type arr = (float, float64_elt, c_layout) Genarray.t

  let arr_of_npy_dir dirname (n : int option) =
  let file_names =
    let a = Sys.readdir dirname in
    match n with
    | None -> a
    | Some k -> Array.sub a 0 k
  in
  let first_file = (Filename.concat dirname (file_names.(0))) in
  let bigarray0 = (bigarray_of_npy first_file) in
  let num_pts = Genarray.nth_dim bigarray0 0 in
  let spatial_dim = Genarray.nth_dim bigarray0 1 in
  let pt_cloud_array =
    Genarray.create Float64 c_layout [| Array.length file_names ; num_pts; spatial_dim |]
  in
  read_dir_i (fun i a -> copy_to i a pt_cloud_array ()) dirname n |>
  Seq.iter (fun () -> ());
  pt_cloud_array;;

  let num_pts_t arr = Genarray.nth_dim arr 0
  let num_spaces arr = Genarray.nth_dim arr 0
  let num_pts_arr arr = Genarray.nth_dim arr 1
  (* let arr_to_bigarray arr = arr *)

  let t_of_npy = bigarray_of_npy;;
  let arr_of_npy filename i =
    let res = Npy.read_mmap3 filename ~shared:false
              |> Npy.to_bigarray3 c_layout Float64
              |> function
              | None -> raise (Invalid_argument "Help")
              | Some bigarray ->
                genarray_of_array3 bigarray
    in
    match i with
    | Some k -> Genarray.sub_left res 0 k
    | None -> res

  let slice_left arr i = Genarray.slice_left arr [| i |];;
  let sub_left arr offset length = Genarray.sub_left arr offset length;;

  let dist v1 v2 =
  let open Genarray in
  assert (num_dims v1 = 1);
  assert (num_dims v2 = 1);
  assert (nth_dim v1 0 = nth_dim v2 0);
  let d = ref 0.0 in
  for i = 0 to (nth_dim v1 0) - 1 do
    let a = (get v1 [| i |] -. (get v2 [| i |])) in
    d := !d +. (a *. a)
  done;
  Float.sqrt !d

  let pdist_helper (m: t) (a: (float, float64_elt, c_layout) Genarray.t) : unit = 
    let open Genarray in 
    let rnm = nth_dim m 0 in
    for i = 0 to rnm - 1 do
      set a [| i; i |] 0.0
    done;
    for i = 0 to rnm - 1 do
      for j = i + 1 to rnm - 1 do
        set a [| i; j |] (dist (slice_left m [| i |]) (slice_left m [| j |]))
        done
    done;
      for i = 0 to rnm - 1 do
        for j = 0 to i - 1 do
          set a [| i ; j |] (get a [| j ; i |])
        done
      done

  let pdist (m: t) : Squareform.t =
  let rnm = (Genarray.dims m).(0) in 
  let open Genarray in 
  let a = create Float64 c_layout [| rnm; rnm |] in
  pdist_helper m a ; Squareform.t_of_bigarray a
  ;;

  let map_pdist(m: arr) : Squareform.arr = 
  let open Genarray in 
  let s = Genarray.dims m in
  let num_pt_clouds = s.(0) in
  let num_pts = s.(1) in
  let a = create Float64 c_layout [| num_pt_clouds; num_pts; num_pts |] in
  for i = 0 to num_pt_clouds - 1 do 
    pdist_helper (Genarray.slice_left m [| i |]) (Genarray.slice_left a [| i |])
  done;
  Squareform.arr_of_bigarray a;;
end

(** pt_cloud_array is of shape (m, n, d), where
    m is the total number of point clouds,
    n is the number of points in each point cloud,
    and d is the dimension of the Euclidean space from which the points are drawn.
*)
