from cajal.qgw import slb_parallel, combined_slb_quantized_gw
import os

def test():
    slb_parallel(
        intracell_csv_loc="tests/icdm.csv",
        out_csv="tests/gw1.csv",
        num_processes=2,
    )
    os.remove("tests/gw1.csv")
    combined_slb_quantized_gw(
        input_icdm_csv_location="tests/icdm.csv",
        gw_out_csv_location="tests/slb_qgw.csv",
        num_processes=2,
        num_clusters=20,
        accuracy=.97,
        nearest_neighbors=3,
        verbose=False,
        chunksize=20
    )
