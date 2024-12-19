Variants of Gromov-Wasserstein
==============================

The Gromov-Wasserstein distance is highly useful for quantifying differences in cell morphology, and a number of variants of Gromov-Wasserstein distance have been proposed in the literature. This page explains two such variants, "Unbalanced Gromov-Wasserstein" and "Fused Gromov-Wasserstein".
Here we focus on the mathematics; for applications see Tutorial 5.

Gromov-Wasserstein
------------------

Before we treat the variants, we will formally define Gromov-Wasserstein distance.

The Gromov-Wasserstein distance deals with cell models of the form :math:`(X,\mu)`, where :math:`X` is a metric space (a set of points throughout the body of the cell, together with a measure of distance between them, which we represent as a square pairwise distance matrix) and :math:`\mu` is a probability measure, a weight function which associates to each point in :math:`X` a non-negative weight such that all weights sum to one.
For neurons, one could set this value proportional to the measured radius of the dendrite at the observed point; for cells with image data, we could set this mass proportional to the pixel intensity of a certain image channel at the given point, measuring the concentration of a protein. In what follows, we will use the uniform probability measure for simplicity.

Given :math:`(X,\mu)` a cell with :math:`n` points and :math:`(Y,\nu)` a cell with :math:`m` points, an :math:`n\times m` matrix :math:`T` with non-negative real entries is a *strict coupling* between :math:`X` and :math:`Y` if :math:`\sum_j T_{ij} = X_i` and :math:`\sum_i T_{ij} = Y_j` for all :math:`i,j`.

For a strict coupling :math:`T`, define

.. math::

   \mathcal{G}(T) = \sum_{ijk\ell}(d^X(x_i,x_j)-d^Y(y_k,y_\ell))^2T_{ik}T_{j\ell}

and

.. math::

   d_{GW}((X,\mu),(Y,\nu)) = \min_T \frac{1}{2} \sqrt{\mathcal{G}(T)}

as :math:`T` ranges over strict couplings.

We call :math:`\mathcal{G}(T)` the Gromov-Wasserstein *cost* of the coupling :math:`T`; the Gromov-Wasserstein *distance* is

.. math::

   d_{GW}((X,\mu),(Y,\nu))=\min_T  \frac{1}{2} \sqrt{\mathcal{G}(T)}

as :math:`T` ranges over strict couplings between :math:`(X,\mu)` and :math:`(Y,\nu)`. (The square root is necessary here in order for :math:`d_{GW}` to satisfy the triangle inequality. The functions in the `CAJAL.run_gw` module report Gromov-Wasserstein *distances* rather than Gromov-Wasserstein *costs* which is inconsistent with some other libraries, such as the Python Optimal Transport library.)

Unbalanced Gromov-Wasserstein
-----------------------------

The big-picture idea behind unbalanced Gromov-Wasserstein is that it is less sensitive to small changes in morphology than ordinary GW. If GW answers the question "How well can we align these two cell morphologies?" then UGW answers the question "How well can we align a large chunk of cell 1 with a large chunk of cell 2, where we want to maximize a weighted sum of the size of the pieces being matched and the goodness-of-fit of the match". In situations where it is safe to discard small pieces of a cell without this substantially changing the morphology, then unbalanced GW might be more robust than GW.

The definition of Gromov-Wasserstein distance involves searching through all possible "couplings" between two cells. The notion of "coupling" employed here is rather rigid and inflexible - cells are regarded as having unit mass, and the couplings are required to satisfy a "conservation of mass" law, that is, all mass in the first cell must be paired with corresponding mass in the second cell. If two neurons are modelled as point clouds with 100 points, then each point will be modelled as having mass 0.01 units, and a valid coupling must satisfy the property that each point in one cell should have 0.01 units worth of mass associated to it from the other cell.

Suppose we have two neurons, which are absolutely identical except that an additional dendrite is present in one which is not present in the other. This would be biologically interesting, and it is plausible that considering such embeddings of one neuron into another would help us to capture important biological similarities. But Gromov-Wasserstein does not recognize such embeddings as valid cell couplings, because it violates "conservation of mass" - all the mass from the first neuron is paired with a fraction of the mass of the second neuron, and the extra dendrite of the other neuron is not paired with anything. The optimal GW transport plan would likely bear no trace of the structural equivalence between the first neuron and a fragment of the second.

The Unbalanced Gromov-Wasserstein distance allows for such embeddings - transport plans which are permitted to create or destroy mass, at the expense of paying a sharp additional penalty cost. The `Unbalanced Gromov-Wasserstein paper <https://arxiv.org/abs/2009.04266>`_ by Séjourné, Vialard, and Peyré provides some useful examples of situations where the extra flexibility of unbalanced Gromov-Wasserstein makes it more tolerant of small differences between objects.


The mathematical framework behind UGW
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The unbalanced Gromov-Wasserstein distance deals with cell models of the form :math:`(X,\mu)`, where :math:`X` is a metric space (a set of points throughout the body of the cell, together with a measure of distance between them, which we represent as a square pairwise distance matrix) and :math:`\mu` is a "measure", a weight function which associates to each point in :math:`X` a non-negative weight. For neurons one could set this value to, for example, the measured radius of the dendrite at the observed point; we will use the uniform probability measure for simplicity.

Formally, given :math:`(X,\mu)` a cell with :math:`n` points and :math:`(Y,\nu)` a cell with :math:`m` points, we define the "unbalanced GW cost" of a transport plan :math:`T \in (\mathbb{R}^+)^{n\times m}` as

.. math::

   \mathcal{L}(T) =  \mathcal{G}(T) + \rho_1 \cdot KL^{\otimes 2}(\pi_X(T)\mid\mu) + \rho_2 \cdot KL^{\otimes 2}(\pi_Y(T)\mid\nu)

and the unbalanced Gromov-Wasserstein distance

.. math::

   UGW((X,\mu),(Y,\nu)) = \min_{T \in (\mathbb{R}^{\geq 0})^{n\times m}} \mathcal{L}(T)

where :math:`\mathcal{G}(T)` was defined above, and

.. math::

   KL(A\mid B) = \sum_{ij}A_{ij}\log\left(\frac{A_{ij}}{B_{ij}}\right) - A_{ij} + B_{ij}

.. math::

   KL^{\otimes 2}(\mu_1\mid \mu_2) = KL(\mu_1\otimes\mu_1\mid \mu_2\otimes\mu_2)

The quantity :math:`\mathcal{G}(T)` captures the direct distortion of the transport plan for the mass that it *does* transport. The Kullback-Leibler divergence :math:`KL` has been generalized here to the case of measures which are not probability measures; its most important properties are that :math:`KL(0\mid B)` is finite and equal to :math:`\sum_{ij} B_{ij}` ; that :math:`KL(\alpha A\mid B)` is a strictly decreasing function of :math:`\alpha` for sufficiently small :math:`\alpha`, with :math:`\frac{\partial KL(\alpha A\mid B)}{\partial \alpha}` approaching :math:`-\infty` as :math:`\alpha\to 0^+`, which guarantees that the minimum of :math:`UGW` is away from :math:`T=0`.

We consider :math:`KL(\pi_X(T)\otimes \pi_X(T)\mid \mu\otimes \mu)` rather than the simpler quantity :math:`KL(\pi_X(T)\mid \mu)` essentially because we want this term to scale in the same way that :math:`\mathcal{G}(T)` does, i.e., quadratically with the magnitude of :math:`T`. More formally, if :math:`\alpha` is a positive scalar then :math:`\mathcal{G}(\alpha T) = \alpha^2\mathcal{G}(T)`, and :math:`KL(\alpha A\mid \alpha B) = \alpha KL(A\mid B)`, so :math:`UGW((X,\alpha \mu),(Y,\alpha \nu)) = \alpha^2UGW((X,\mu),(Y,\nu))`. Thus, the *relative* UGW distances between spaces (expressed as a ratio) are independent of the unit of "mass" chosen.

We refer to :math:`\pi_X(T), \pi_Y(T)` as the "marginals" of :math:`T` and refer to :math:`\rho_1KL(...)+\rho_2KL(...)` as the marginal penalty term.

Interpretation of :math:`\rho_1,\rho_2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The coefficients :math:`\rho_1, \rho_2` are chosen by the experimenter to reflect the relative tradeoff they want to impose between distortion and the creation/destruction of mass. We can illustrate some corner cases to give a feel for the behavior.

* When :math:`\rho_1,\rho_2` are chosen to be very large, approaching :math:`\infty`, then :math:`UGW((X,\mu),(Y,\nu)) \approx GW((X,\mu),(Y,\nu))`.
* If :math:`\rho_1` is set to zero and :math:`\rho_2` is chosen close to infinity, then the algorithm is free to throw away or create as much mass from :math:`\mu` as it wants, and so it will search for the closest thing to an isometric embedding of :math:`Y` into :math:`X` (the masses no longer play a role); if there exists an isometric embedding of :math:`Y` into :math:`X`, then the unbalanced GW distance will be zero.
* As :math:`\rho_1,\rho_2` decrease, then the total mass destroyed by the transport plan will monotonically increase, as the algorithm can always decrease the distortion of the transported mass by simply transporting less mass, and reducing :math:`\rho_1` and :math:`\rho_2` makes it cheaper to do that.
* UGW and strict GW are directly comparable and :math:`UGW \leq GW`, because every "balanced" transport plan is also an "unbalanced" transport plan. (We can *not* guarantee that this inequality will be empirically observed in computational results, because all these algorithms are only giving us approximations to GW and UGW by upper bound, not the actual quantity.)
* If :math:`\rho_1=\rho_2` then :math:`UGW` is a symmetric function of its arguments, which is appropriate in situations where you want to think of it as defining a "morphology space." Formally, UGW does not form a metric, so the term "morphology space" is optimistic.

Choosing :math:`\rho_1,\rho_2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To choose :math:`\rho_1,\rho_2` appropriately, some amount of experimentation is necessary, but one can compute the ordinary GW distances between cells beforehand to get an idea of the appropriate coefficients.

The following analysis may help to choose :math:`\rho_1,\rho_2`. Let :math:`\mu,\nu` be measures, and :math:`T` any coupling. Then some elementary calculus shows that the optimal rescaling of :math:`T` (the overall fraction of mass that should be kept by the transport plan :math:`T` to minimize the UGW cost) is

.. math::

   \operatorname{argmin}_{\alpha} \mathcal{L}(\alpha T) = \exp\left(\frac{-(\mathcal{L}(T) + \rho_1(m(T)^2-m(\mu)^2) +\rho_2(m(T)^2-m(\nu)^2)))}{2m(T)^2(\rho_1+\rho_2)}\right)

This shows that for any equilibrium solution to the UGW problem,

.. math::

   m(T)^2 = \frac{\rho_1m(\mu)^2 + \rho_2m(\nu)^2 - \mathcal{L}(T)}{\rho_1+\rho_2}

and that the ratio :math:`\frac{\mathcal{L}(T)}{\rho_1+\rho_2}` must be controlled in order to bound the mass lost by the transport plan. It also shows that the lower the cost of the transport plan :math:`\mathcal{L}(T)`, the less mass will be lost overall.

In particular, if :math:`T` is any strict coupling between probability distributions :math:`\mu` and :math:`\nu` (the solution to the GW transport problem) then :math:`\alpha = \exp{\frac{-\mathcal{G}(T)}{2(\rho_1+\rho_2)}}` is
the optimal rescaling of :math:`T` for the unbalanced GW problem, which gives an upper bound of :math:`(\rho_1+\rho_2)(1-\exp{\frac{-\mathcal{G}(T)}{\rho_1+\rho_2}})` for UGW. (This approaches :math:`\mathcal{G}(T)` asymptotically from below as :math:`\rho_1+\rho_2\to\infty`, as can be seen by L'Hospital's rule.)

..
   To understand this expression, it is helpful to note that :math:`e^x\approx (1+x/a)^a` for :math:`a>>x`); thus :math:`e^{x/a}\approx (1+ x/a)` for :math:`a>>x`,
   and so :math:`(\rho_1+\rho_2)(1-\exp{\frac{-\mathcal{G}(T)}{\rho_1+\rho_2}}) \approx (\rho_1+\rho_2)(1-(1-\frac{\mathcal{G}(T)}{\rho_1+\rho_2})) = \mathcal{G}(T)` for :math:`\rho_1+\rho_2>>\mathcal{G}(T)`.

So, if you do not want more than 10% of the mass of the cells to be destroyed by the transport plan, you should choose :math:`\rho_1,\rho_2` such that :math:`\rho_1+\rho_2\geq -\mathcal{G}(T)/2\ln(0.9)`.

:math:`d_{GW}((A,\mu),(B,\nu))` is 50 units (so that :math:`\mathcal{G}(T)` = 10000, for the optimal strict coupling :math:`T`)
then for given values of :math:`\rho_1,\rho_2`, an upper bound for :math:`UGW((A,\mu),(B,\nu))` is :math:`\min_{\alpha\in[0,1]} \mathcal{L}(\alpha T)= 10000 \cdot \min_{\alpha\in[0,1]} \alpha^2 + (\rho_1+\rho_2)\sigma(\alpha^2)`, where :math:`\sigma(x)= x\ln x-x+ 1`.


If :math:`(A,\mu)` and :math:`(B,\nu)` are two cells with unit mass (i.e., :math:`\mu` and :math:`\nu` are probability distributions) and the Gromov-Wasserstein distance :math:`d_{GW}((A,\mu),(B,\nu))` is 50 units (so that :math:`\mathcal{G}(T)` = 10000, for the optimal strict coupling :math:`T`)
then for given values of :math:`\rho_1,\rho_2`, an upper bound for :math:`UGW((A,\mu),(B,\nu))` is :math:`\min_{\alpha\in[0,1]} \mathcal{L}(\alpha T)= 10000 \cdot \min_{\alpha\in[0,1]} \alpha^2 + (\rho_1+\rho_2)\sigma(\alpha^2)`, where :math:`\sigma(x)= x\ln x-x+ 1`.

Computational complications
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unbalanced GW is computed using a different algorithm than ordinary GW, and it requires an extra parameter to guide the algorithm, which we now describe.

Define

.. math::

   \mathcal{L}_{\varepsilon}(T) = \mathcal{L}(T) + \varepsilon \cdot KL^{\otimes 2}(T \mid \mu\otimes\nu)

and

.. math::

   UGW_\varepsilon((X,\mu),(Y,\nu)) = \min_{T\in(\mathbb{R}^{\geq 0})^{n\times m}} \mathcal{L}_\varepsilon(T)

We call :math:`\mathcal{L}_{\varepsilon}(T)` the "entropically regularized cost function". The most important property of :math:`\mathcal{L}_\varepsilon` is that :math:`\lim_{T_{ij}\to 0^+}\frac{\partial \mathcal{L}(T)}{\partial T_{ij}}\bigg\rvert_{T_{ij=0}}= -\infty`, and for :math:`T` with :math:`T_{ij}=0`, we can always reduce :math:`\mathcal{L}_{\varepsilon}(T)` by increasing :math:`T_{ij}` to some very small :math:`\epsilon`. This implies that the global minimum of :math:`\mathcal{L}_\varepsilon(T)` lies in :math:`(\mathbb{R}^{>0})^{n\times m}`, the strict interior of :math:`(\mathbb{R}^{\geq 0})^{n\times m}`, which lets us apply techniques based on calculus. Therefore, we try to solve this problem instead; we know that in the limit as :math:`\varepsilon\to 0^+`, :math:`UGW_\varepsilon\to UGW`.

The coefficient :math:`\varepsilon` has the physical meaning that the 'couplings' will smear each point in the first cell across *every* point in the second cell, at least to some small degree; a bit like an electron cloud, when a point from :math:`X` is transferred across the coupling to :math:`Y` it may be highly localized in a certain region of :math:`Y`, but it has nonzero probability mass everywhere in :math:`Y`.

To choose :math:`\varepsilon` appropriately, we advise that you experiment with your data set at different values of :math:`\varepsilon`. For sufficiently small :math:`\varepsilon`, the algorithm will tend to diverge due to numerical instability; the most accurate possible result will be given by choosing the smallest possible value of :math:`\varepsilon` for which all values terminate. One can increase :math:`\varepsilon` beyond this point, which will tend to make the algorithm converge faster, but to a less accurate answer, so it is a tradeoff to be made based on the size of the dataset to be computed and the precision necessary for the results to be useful. Our experience is that :math:`\rho_1,\rho_2` should be at least 20x larger than :math:`\varepsilon` for the results to be decently accurate, and a higher ratio is probably better.

Fused Gromov-Wasserstein
------------------------

We let $(X,\mu)$, $(Y,\nu)$ and $\mathcal{G}$ be as before.

Classical Gromov-Wasserstein treats cells purely geometrically, as shapes. In searching for good alignments between two neurons, it doesn't consider some important information present in cell morphology reconstructions, such as the labels for the soma and dendrite nodes.
On biological grounds, it is reasonable to argue that a "good alignment" between two neurons should align the soma node to the soma node, align axon to axon, basal dendrites to basal dendrites, and apical dendrites to apical dendrites. Fused Gromov-Wasserstein is a construct
which modifies classical Gromov-Wasserstein to add a penalty term for transport plans which align nodes of different types. By making the penalty term large, we can bias the search algorithm towards transport plans which reflect the additional information available in the cell structure.

The formula for the fused GW cost of a transport plan is

.. math::

   \mathcal{F}(T) = \alpha\mathcal{G}(T) + (1-\alpha)\sum_{ij}C_{ij}T_{ij}

and we define

.. math::
   FGW_C((X,\mu),(Y,\nu)) = \inf_T \mathcal{F}(T)

where $C_{ij}$ is a user-supplied penalty matrix, and the value $C_{ij}$ indicates the intrinsic penalty for aligning $X_i$ to $Y_j$.

In our implementation, the user supplies the penalty for aligning nodes of distinct SWC structure id labels. It is easiest to choose these on relative grounds: for example, if the user wants to impose the constraint that aligning a soma node to a dendrite node is ten times worse than
aligning a basal dendrite node to an apical node, they can choose the soma-to-dendrite penalty to be 10 and the basal-to-apical penalty to be 1. Once this is done, it remains to choose the coefficient $\alpha$ appropriately.

Note that if $T^{GW}$ is the optimal transport plan for classical Gromov-Wasserstein and $C$ is a proposed cost matrix, then an upper bound for $FGW_C(X,Y)$ is $\mathcal{F}(T^{GW})$.
It follows that, if $T^{FGW}$ is the optimal transport plan for fused GW, then

.. math::
   \mathcal{G}(T^{FGW}) \leq \mathcal{G}(T^{GW}) + (\frac{1-\alpha}{\alpha})(\sum_{ij}C_{ij}T^{GW}_{ij})

One can interpret this inequality as follows: by increasing the term $(\frac{1-\alpha}{\alpha})$, the algorithm will be willing to accept higher distortion in order to better align nodes of similar types. If the user chooses, say, $(\frac{1-\alpha}{\alpha}) = 0.3$,
then the GW cost of the transport plan $T^{FGW}$ will be at most 30% more than the GW cost of the original transport plan. Thus, our approach to giving an interpretable interface is to allow the user to control how much additional distortion they are willing to accept in the transport plan
in order to align nodes of the same type.


