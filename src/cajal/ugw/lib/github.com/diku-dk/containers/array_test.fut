import "../sorts/radix_sort"
import "../segmented/segmented"
import "array"

local def count_occourences_sort [n] (arr: [n]i32): [](i32, i32) =
  let sorted = radix_sort_int i32.num_bits i32.get_bit arr
  let flags =
    map (
      \i ->
        i == 0 || sorted[i - 1] != sorted[i]
    ) (iota n)
  let as = segmented_scan (+) 0 flags (replicate n 1) |> zip sorted
  let segment_ends = rotate 1 flags
  let segment_end_offsets = segment_ends |> map i64.bool |> scan (+) 0
  let num_segments = if n > 0 then last segment_end_offsets else 0
  let scratch = replicate num_segments (0, 0)
  let index i f = if f then i-1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as

local def dedup_sort [n] (arr: [n]i32): []i32 =
  let sorted = radix_sort_int i32.num_bits i32.get_bit arr
  let flags =
    map (
      \i ->
        i == 0 || sorted[i - 1] != sorted[i]
    ) (iota n)
  in zip flags sorted
     |> filter (.0)
     |> map (.1)
  
local def hash_i32 a x = hash_i64 a (i64.i32 x)
  
local def count_occourences [n] (arr: [n]i32): [](i32, i32) =
  reduce_by_key hash_i32 (==) 0i32 (+) <| map (\a -> (a, 1)) arr

-- ==
-- entry: test_reduce_by_key
-- compiled random input { [100][100]i32 }
-- output { true }
-- compiled random input { [100][5]i32 }
-- output { true }
entry test_reduce_by_key [n][m] (arrs: [n][m]i32): bool =
  all (
    \arr ->
      let arr = map (% 10) arr
      let sort_counts = count_occourences_sort arr
      let size = length sort_counts
      let sort_counts = sized size sort_counts
      let counts =
        count_occourences arr
        |> radix_sort_int_by_key (.0) i32.num_bits i32.get_bit
        |> sized size
      in map2 (==) sort_counts counts |> and
  ) arrs

-- ==
-- entry: test_dedup
-- compiled random input { [100][100]i32 }
-- output { true }
-- compiled random input { [100][5]i32 }
-- output { true }
entry test_dedup [n][m] (arrs: [n][m]i32): bool =
  all (
    \arr ->
      let arr = map (% 10) arr
      let sort_dedups = dedup_sort arr
      let size = length sort_dedups
      let sort_dedups = sized size sort_dedups
      let counts =
        dedup hash_i32 (==) arr
        |> radix_sort_int i32.num_bits i32.get_bit
        |> sized size
      in map2 (==) sort_dedups counts |> and
  ) arrs
