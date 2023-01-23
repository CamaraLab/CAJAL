Running Gromov-Wasserstein
==========================

The GW distance is calculated using the same function whether the distance matrices represent the Euclidean or geodesic metric.

A representative TinyDB \*.json database is structured as:

.. code-block::  javascript

	{"_default":
	      {"1": {"name": "epd210cmdil3_1_0",
                     "cell": [5.0990195135927845,
		              9.219544457292887,
			      14.317821063276353,
			      19.6468827043885,
			      26.086394921491163,
			      29.410882339705484,
			      32.8709598277872,
		              // this list contains 1,225 = 49 * 50/2 entries
			      // corresponding to a cell with 50 sample points
			      ] },
               "2": {"name": "epd210cmdil3_1_1",
                     "cell": [11.045361017187261,
                              22.090722034374522,
			      33.13608305156178,
		              // (...)
			      ] }, // (...)
	       }
	}	     
  

.. autofunction:: run_gw.compute_gw_distance_matrix

