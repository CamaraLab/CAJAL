from cajal.qgw import slb_parallel
from cajal.combined_slb_qgw import combined_slb_quantized_gw
import os


def test_slb_parallel():
    slb_parallel(
        intracell_csv_loc="tests/icdm.csv",
        out_csv="tests/gw1.csv",
        num_processes=2,
    )
    os.remove("tests/gw1.csv")


def test_combined_slb_quantized_gw():
    combined_slb_quantized_gw(
        input_icdm_csv_location="tests/icdm.csv",
        gw_out_csv_location="tests/slb_qgw.csv",
        num_processes=2,
        num_clusters=20,
        accuracy=0.97,
        nearest_neighbors=3,
        verbose=False,
        chunksize=20,
    )
