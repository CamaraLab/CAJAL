# Functions to fix up SWC formatting to meet assumptions of various software
# General SWC format assumed to be http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
# The one assumption CAJAL makes that we don't fix here is that parent nodes must come before any of their children
import os
import copy
import numpy as np
import warnings

from CAJAL.lib.run_gw import pj


def set_index(swc, new_index=1):
    """
    Shifts SWC node IDs to new minimum index (i.e. from zero-index to one-index, required by some external software)

    Args:
        swc (numpy array): matrix of rows in SWC format
        new_index (integer): new minimum node ID

    Returns:
        reformatted swc numpy array
    """
    new_index = int(new_index)
    new_swc = copy.deepcopy(swc)
    old_index = np.min(new_swc[new_swc[:, 0] != -1, 0])
    offset = new_index - old_index
    new_swc[:, 0] = new_swc[:, 0] + offset
    new_swc[new_swc[:, -1] != -1, -1] = new_swc[new_swc[:, -1] != -1, -1] + offset
    return new_swc


def make_sequential(swc):
    """
    Renumbers SWC node IDs to be sequential (required by some external software, such as TMD)

    Args:
        swc (numpy array): matrix of rows in SWC format

    Returns:
        reformatted swc numpy array
    """
    new_swc = copy.deepcopy(swc)
    new_id_dict = {-1: -1}
    for i in range(len(new_swc)):
        new_id_dict[new_swc[i, 0]] = i + 1
    new_swc[:, 0] = [new_id_dict[x] for x in new_swc[:, 0]]
    new_swc[:, -1] = [new_id_dict[x] for x in new_swc[:, -1]]
    return new_swc


def rm_disconnect(swc):
    """
    Removes branches not connected to soma (required by some external software, such as NBLAST & TMD)

    Args:
        swc (numpy array): matrix of rows in SWC format

    Returns:
        reformatted swc numpy array
    """
    new_swc = copy.deepcopy(swc)
    # No, I can't just remove all after the first -1 parent. Some have connected components after that.
    # But nodes are ordered so parent always comes first, so can loop through and remove if parent is to be removed
    remove_node = list(new_swc[np.where(new_swc[:, -1] == -1)[0][1:], 0])
    for row in range(1, new_swc.shape[0]):
        if new_swc[row, -1] in remove_node:
            remove_node.append(new_swc[row, 0])
    new_swc = new_swc[np.logical_not(np.isin(new_swc[:, 0], remove_node))]
    return new_swc


def add_root(swc):
    """
    Adds dummy root node before soma (required by some external software, such as NBLAST)

    Args:
        swc (numpy array): matrix of rows in SWC format

    Returns:
        reformatted swc numpy array
    """
    new_swc = copy.deepcopy(swc)
    new_root = copy.deepcopy(swc[0])
    new_swc[0, -1] = new_root[0] - 1  # so when add 1 will be new_root ID
    new_swc[:, 0] = new_swc[:, 0] + 1
    new_swc[:, -1] = new_swc[:, -1] + 1
    new_swc = np.r_[new_root.reshape((1, 7)), new_swc]
    return new_swc


def reformat_swc_file(infile, outfile, new_index=1, sequential=True,
                      rmdisconnect=False, dummy_root=False, keep_header=True):
    """
    Read in SWC file and apply reformatting functions, keep header

    Args:
        infile (string): path to SWC file to reformat
        outfile (string): path to save reformatted SWC
        new_index (integer, or None): if integer, will shift SWC node IDs to use this as the minimum. If None, skip.
        sequential (boolean): rename SWC node IDs to be sequential 1:N (or starting at whichever minimum index)
        rmdisconnect (boolean): remove branches not connected to soma (these start with a non-soma node with parent -1)
        dummy_root (boolean): add dummy node at same coordinate as soma, set as parent of soma
        keep_header (boolean): keep header (first few lines starting with #) when re-saving SWC file

    Returns:
        None (writes to file)
    """
    if infile[-4:] != ".SWC" and infile[-4:] != ".swc":
        warnings.warn("{} is not a .SWC or .swc file, skipping".format(infile))
        return

    swc = np.loadtxt(infile)
    if new_index is not None:
        swc = set_index(swc, new_index)
    if rmdisconnect:
        swc = rm_disconnect(swc)
        sequential = True  # because potentially removing nodes from middle, need to reorder
    if dummy_root:
        swc = add_root(swc)
    if sequential:
        swc = make_sequential(swc)

    # Get the header
    if keep_header:
        with open(infile) as f:
            header = []
            for line in f:
                if line[0] == "#":
                    header.append(line)
                else:
                    break
    else:
        header = []

    with open(outfile, "w+") as f:
        for line in header:
            f.write(line)
        for row in range(swc.shape[0]):
            f.write(" ".join([str(int(swc[row, i])) if i in [0, 1, 6]
                              else str(swc[row, i]) for i in range(swc.shape[1])]) + "\n")


def bulk_reformat_swc_file(infolder, outfolder, new_index=1, sequential=True,
                           rmdisconnect=False, dummy_root=False, keep_header=True):
    """
    Apply same reformatting to each SWC file in a folder, resave to a new folder

    Args:
        infolder (string): path to folder containing SWC files to reformat
        outfolder (string): path to folder in which to save reformatted SWCs
        new_index (integer, or None): if integer, will shift SWC node IDs to use this as the minimum. If None, skip.
        sequential (boolean): rename SWC node IDs to be sequential 1:N (or starting at whichever minimum index)
        rmdisconnect (boolean): remove branches not connected to soma (these start with a non-soma node with parent -1)
        dummy_root (boolean): add dummy node at same coordinate as soma, set as parent of soma
        keep_header (boolean): keep header (first few lines starting with #) when re-saving SWC file

    Returns:
        None (writes to files)
    """
    for swc_file in os.listdir(infolder):
        reformat_swc_file(pj(infolder, swc_file), pj(outfolder, swc_file),
                          new_index, sequential, rmdisconnect, dummy_root, keep_header)
