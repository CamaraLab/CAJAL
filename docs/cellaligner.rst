Analysis of Subcellular Organization
====================================

Quantitative analysis of subcellular protein organization is often confounded by variation in cell morphology, limiting 
the identification and interpretation of localization patterns in fluorescence microscopy data from morphologically 
complex cells, such as neurons and glia. To address this gap, CAJAL includes CellAligner, a general framework for 
quantifying and comparing subcellular organization in 2D imaging data while explicitly accounting for differences 
in cell morphology. CellAligner takes as input multi-channel images, including fluorescently labeled proteins and 
morphological stains (e.g., nuclear, membrane, or organelle stains), along with cell segmentation masks. To quantify 
differences in subcellular localization independently of morphology, CellAligner first maps all cells to a shared 
anchor cell by computing a pixel-wise fused unbalanced GW mapping. This mapping jointly minimizes discrepancies in 
morphological staining intensities and intracellular distances between each cell and the anchor, and it allows partial 
matching through unbalanced regularization so that cells with substantially different geometries can be partially mapped. 
The resulting mapping is used to transfer each cell's protein distribution onto the morphology of the anchor cell. 
Differences in subcellular protein localization are then quantified by computing standard colocalization metrics between 
the mapped protein distributions within the anchor geometry. CellAligner-OT uses optimal transport distances, due to 
their natural interpretation as the average intracellular distance over which mass from one protein distribution must 
be transported to match another. Alternatively, existing cell image analysis methods, such as `CellProfiler <https://cellprofiler.org/>`_, 
`Cytoself <https://github.com/royerlab/cytoself>`_, and `Paired Cell Inpainting <https://github.com/alexxijielu/paired_cell_inpainting>`_, 
can be applied to the corresponding anchor-cell representations, thereby reducing morphological-associated confounding from downstream analyses. 

Repeating the mapping procedure across several morphologically distinct anchors and integrating the resulting distances 
using a non-coordinate extension of the weighted nearest-neighbor algorithm28 yields subcellular localization distances 
that are robust to differences in cell morphology. Thus, in addition to a cell morphology summary space, CellAligner 
produces a subcellular localization summary space in which each point represents a cell and pairwise distances reflect 
differences in protein localization independent of cell morphology. These localization summary spaces can be integrated 
across proteins to produce higher-level subcellular organization spaces, in which pairwise distances summarize 
differences in intracellular organization between cells.

CellAligner involves solving an expensive computational optimization problem, which limits its scalability to large 
imaging datasets. To enable CellAligner-based analysis in high-throughput settings, CAJAL also includes deep CellAligner-OT 
(dCellAligner-OT), a deep metric learning framework that efficiently approximates CellAligner-OT subcellular protein 
localization distances and anchor-cell representations of protein distributions. The dCellAligner-OT architecture 
consists of a convolutional EfficientNet encoder that takes as input the cellular and nuclear segmentation masks and 
the protein-staining image for each cell, together with a U-Net decoder that produces an anchor-cell representation 
of the protein distribution that can be analyzed using existing cell image analysis methods. The model is trained 
using a two-stage procedure. In the first stage, the model is optimized using a reconstruction objective that 
encourages the embeddings to retain the information necessary to reconstruct the mapped protein distribution in the 
anchor cell. In the second stage, the model is fine-tuned as a Siamese network using an additional metric-learning 
loss that penalizes discrepancies between CellAligner-OT distances computed for pairs of cells and Euclidean distances 
between their corresponding embeddings. Thus, in the fully trained model, Euclidean distances in the low-dimensional 
dCellAligner-OT embedding approximate the corresponding CellAligner-OT distances between pairs of cells. In addition, 
the metric-learning stage regularizes the anchor-cell reconstructions, improving the quality of the mapped representations.

Additional information about CellAligner can be found at:

\- Hu, R., et el. `Morphology-robust quatification of subcellular organization in complex cells. <https://www.biorxiv.org/content/10.64898/2026.05.28.728543>`_ bioRxiv (2026).