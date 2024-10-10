type A = #left | #right

def boolean_filter [n] 'b (a : [n]bool) (b0 : [n]b) (b1 : [n]b) =
tabulate n (\i -> if a[i] then b0[i] else b1[i]) 

def if_then_else 'b (a : A) (b0: b) (b1: b) =
  match a
  case #left -> b0
  case #right -> b1

-- def left_right_filter[n] 'b (a : [n]A) (b : [n]B) (c : [n]C) =
--   tabulate (\i -> match a[i]
--    		#left -> b[i]
-- 		#right -> c[i]
-- 	   ) n
