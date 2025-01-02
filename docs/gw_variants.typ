= Variants of Gromov-Wasserstein

The Gromov-Wasserstein distance is highly useful for quantifying differences in cell morphology, and a number of variants of Gromov-Wasserstein distance have been proposed in the literature. This page explains two such variants, "Unbalanced Gromov-Wasserstein" and "Fused Gromov-Wasserstein".
Here we focus on the mathematics; for applications see Tutorial 5.

== Gromov-Wasserstein

Before we treat the variants, we will formally define Gromov-Wasserstein distance.

#let GW = [GW]
#let KL2(x,y) = $[K L]^(times.circle 2)(x mid y)$
#let UGW = [U G W]

The Gromov-Wasserstein distance deals with cell models of the form $(X,mu)$, where $X$ is a metric space (a set of points throughout the body of the cell, together with a measure of distance between them, which we represent as a square pairwise distance matrix) and $mu$ is a probability measure, a weight function which associates to each point in $X$ a non-negative weight such that all weights sum to one.
For neurons, one could set this value proportional to the measured radius of the dendrite at the observed point; for cells with image data, we could set this mass proportional to the pixel intensity of a certain image channel at the given point, measuring the concentration of a protein. In what follows, we will use the uniform probability measure for simplicity.

Given $(X,mu)$ a cell with $n$ points and $(Y,nu)$ a cell with $m$ points, an $n times m$ matrix $T$ with non-negative real entries is a *strict coupling* between $X$ and $Y$ if $sum_j T_(i j) = X_i$ and $sum_i T_(i j) = Y_j$ for all $i,j$.

For a strict coupling $T$, define

$cal(G)(T) = sum_(i j k ell)(d^X(x_i,x_j)-d^Y(y_k,y_ell))^2T_(i k)T_(j ell)$

and

$d_(GW)((X,mu),(Y,nu)) = min_T frac(1,2) sqrt(cal(G)(T))$

as $T$ ranges over strict couplings.

We call $cal(G)(T)$ the Gromov-Wasserstein *cost* of the coupling $T$; the Gromov-Wasserstein *distance* is

$d_(GW)((X,mu),(Y,nu))= min_T frac(1,2) sqrt(cal(G)(T))$

as $T$ ranges over strict couplings between $(X,mu)$ and $(Y,nu)$. (The square root is necessary here in order for $d_(GW)$ to satisfy the triangle inequality. The functions in the `CAJAL.run_gw` module report Gromov-Wasserstein *distances* rather than Gromov-Wasserstein *costs* which is inconsistent with some other libraries, such as the Python Optimal Transport library.)

== Unbalanced Gromov-Wasserstein

The big-picture idea behind unbalanced Gromov-Wasserstein is that it is less sensitive to small changes in morphology than ordinary GW. If GW answers the question "How well can we align these two cell morphologies?" then UGW answers the question "How well can we align a large chunk of cell 1 with a large chunk of cell 2, where we want to maximize a weighted sum of the size of the pieces being matched and the goodness-of-fit of the match". In situations where it is safe to discard small pieces of a cell without this substantially changing the morphology, then unbalanced GW might be more robust than GW.

The definition of Gromov-Wasserstein distance involves searching through all possible "couplings" between two cells. The notion of "coupling" employed here is rather rigid and inflexible - cells are regarded as having unit mass, and the couplings are required to satisfy a "conservation of mass" law, that is, all mass in the first cell must be paired with corresponding mass in the second cell. If two neurons are modelled as point clouds with 100 points, then each point will be modelled as having mass 0.01 units, and a valid coupling must satisfy the property that each point in one cell should have 0.01 units worth of mass associated to it from the other cell.

Suppose we have two neurons, which are absolutely identical except that an additional dendrite is present in one which is not present in the other. This would be biologically interesting, and it is plausible that considering such embeddings of one neuron into another would help us to capture important biological similarities. But Gromov-Wasserstein does not recognize such embeddings as valid cell couplings, because it violates "conservation of mass" - all the mass from the first neuron is paired with a fraction of the mass of the second neuron, and the extra dendrite of the other neuron is not paired with anything. The optimal GW transport plan would likely bear no trace of the structural equivalence between the first neuron and a fragment of the second.

The Unbalanced Gromov-Wasserstein distance allows for such embeddings - transport plans which are permitted to create or destroy mass, at the expense of paying a sharp additional penalty cost. The #link("https://arxiv.org/abs/2009.04266")[Unbalanced Gromov-Wasserstein paper] by Séjourné, Vialard, and Peyré provides some useful examples of situations where the extra flexibility of unbalanced Gromov-Wasserstein makes it more tolerant of small differences between objects.


== The mathematical framework behind UGW

The unbalanced Gromov-Wasserstein distance deals with cell models of the form $(X,mu)$, where $X$ is a metric space (a set of points throughout the body of the cell, together with a measure of distance between them, which we represent as a square pairwise distance matrix) and $mu$ is a "measure", a weight function which associates to each point in $X$ a non-negative weight. For neurons one could set this value to, for example, the measured radius of the dendrite at the observed point; we will use the uniform probability measure for simplicity.

Formally, given $(X,mu)$ a cell with $n$ points and $(Y,nu)$ a cell with $m$ points, we define the "unbalanced GW cost" of a transport plan $T in (bb(R)^+)^(n times m)$ as
$cal(L)(T) = cal(G)(T) + rho_1 circle.stroked.tiny KL2(pi_X(T), mu) + rho_2 circle.stroked.tiny KL2(pi_Y(T),nu)$

and the unbalanced Gromov-Wasserstein distance

$UGW((X,mu),(Y,nu)) = min_(T in (bb(R)^(<= 0))^(n times m))cal(L)(T)$

// where $cal(G)(T)$ was defined above, and

// $KL(A\mid B) = \sum_{ij}A_{ij}\log\left(\frac{A_{ij}}{B_{ij}}\right) - A_{ij} + B_{ij}$

// $KL^{otimes 2}(\mu_1\mid \mu_2) = KL(\mu_1otimes\mu_1\mid \mu_2otimes\mu_2)$

// The quantity $cal(G)(T)$ captures the direct distortion of the transport plan for the mass that it *does* transport. The Kullback-Leibler divergence $KL$ has been generalized here to the case of measures which are not probability measures; its most important properties are that $KL(0\mid B)$ is finite and equal to $\sum_{ij} B_{ij}$ ; that $KL(\alpha A\mid B)$ is a strictly decreasing function of $\alpha$ for sufficiently small $\alpha$, with $\frac{\partial KL(\alpha A\mid B)}{\partial \alpha}$ approaching $-infty$ as $\alpha\to 0^+$, which guarantees that the minimum of $UGW$ is away from $T=0$.

// We consider $KL(pi_X(T)otimes pi_X(T)\mid \muotimes \mu)$ rather than the simpler quantity $KL(pi_X(T)\mid \mu)$ essentially because we want this term to scale in the same way that $cal(G)(T)$ does, i.e., quadratically with the magnitude of $T$. More formally, if $\alpha$ is a positive scalar then $cal(G)(\alpha T) = \alpha^2cal(G)(T)$, and $KL(\alpha A\mid \alpha B) = \alpha KL(A\mid B)$, so $UGW((X,\alpha \mu),(Y,\alpha \nu)) = \alpha^2UGW((X,\mu),(Y,\nu))$. Thus, the *relative* UGW distances between spaces (expressed as a ratio) are independent of the unit of "mass" chosen.

// We refer to $pi_X(T), pi_Y(T)$ as the "marginals" of $T$ and refer to $rho_1KL(...)+rho_2KL(...)$ as the marginal penalty term.

// Interpretation of $rho_1,rho_2$
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// The coefficients $rho_1, rho_2$ are chosen by the experimenter to reflect the relative tradeoff they want to impose between distortion and the creation/destruction of mass. We can illustrate some corner cases to give a feel for the behavior.

// * When $rho_1,rho_2$ are chosen to be very large, approaching $infty$, then $UGW((X,\mu),(Y,\nu)) \approx GW((X,\mu),(Y,\nu))$.
// * If $rho_1$ is set to zero and $rho_2$ is chosen close to infinity, then the algorithm is free to throw away or create as much mass from $\mu$ as it wants, and so it will search for the closest thing to an isometric embedding of $Y$ into $X$ (the masses no longer play a role); if there exists an isometric embedding of $Y$ into $X$, then the unbalanced GW distance will be zero.
// * As $rho_1,rho_2$ decrease, then the total mass destroyed by the transport plan will monotonically increase, as the algorithm can always decrease the distortion of the transported mass by simply transporting less mass, and reducing $rho_1$ and $rho_2$ makes it cheaper to do that.
// * UGW and strict GW are directly comparable and $UGW \leq GW$, because every "balanced" transport plan is also an "unbalanced" transport plan. (We can *not* guarantee that this inequality will be empirically observed in computational results, because all these algorithms are only giving us approximations to GW and UGW by upper bound, not the actual quantity.)
// * If $rho_1=rho_2$ then $UGW$ is a symmetric function of its arguments, which is appropriate in situations where you want to think of it as defining a "morphology space." Formally, UGW does not form a metric, so the term "morphology space" is optimistic.

// Choosing $rho_1,rho_2$
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// To choose $rho_1,rho_2$ appropriately, some amount of experimentation is necessary, but one can compute the ordinary GW distances between cells beforehand to get an idea of the appropriate coefficients.

// The following analysis may help to choose $rho_1,rho_2$. Let $\mu,\nu$ be measures, and $T$ any coupling. Then some elementary calculus shows that the optimal rescaling of $T$ (the overall fraction of mass that should be kept by the transport plan $T$ to minimize the UGW cost) is

// $\operatorname{argmin}_{\alpha} cal(L)(\alpha T) = \exp\left(\frac{-(cal(L)(T) + rho_1(m(T)^2-m(\mu)^2) +rho_2(m(T)^2-m(\nu)^2)))}{2m(T)^2(rho_1+rho_2)}\right)$

// This shows that for any equilibrium solution to the UGW problem,

// $m(T)^2 = \frac{rho_1m(\mu)^2 + rho_2m(\nu)^2 - cal(L)(T)}{rho_1+rho_2}$

// and that the ratio $\frac{cal(L)(T)}{rho_1+rho_2}$ must be controlled in order to bound the mass lost by the transport plan. It also shows that the lower the cost of the transport plan $cal(L)(T)$, the less mass will be lost overall.

// In particular, if $T$ is any strict coupling between probability distributions $\mu$ and $\nu$ (the solution to the GW transport problem) then $\alpha = \exp{\frac{-cal(G)(T)}{2(rho_1+rho_2)}}$ is
// the optimal rescaling of $T$ for the unbalanced GW problem, which gives an upper bound of $(rho_1+rho_2)(1-\exp{\frac{-cal(G)(T)}{rho_1+rho_2}})$ for UGW. (This approaches $cal(G)(T)$ asymptotically from below as $rho_1+rho_2\toinfty$, as can be seen by L'Hospital's rule.)

// ..
//    To understand this expression, it is helpful to note that $e^x\approx (1+x/a)^a$ for $a>>x$); thus $e^{x/a}\approx (1+ x/a)$ for $a>>x$,
//    and so $(rho_1+rho_2)(1-\exp{\frac{-cal(G)(T)}{rho_1+rho_2}}) \approx (rho_1+rho_2)(1-(1-\frac{cal(G)(T)}{rho_1+rho_2})) = cal(G)(T)$ for $rho_1+rho_2>>cal(G)(T)$.

// So, if you do not want more than 10% of the mass of the cells to be destroyed by the transport plan, you should choose $rho_1,rho_2$ such that $rho_1+rho_2<= -cal(G)(T)/2\ln(0.9)$.

// $d_(GW)((A,\mu),(B,\nu))$ is 50 units (so that $cal(G)(T)$ = 10000, for the optimal strict coupling $T$)
// then for given values of $rho_1,rho_2$, an upper bound for $UGW((A,\mu),(B,\nu))$ is $min_{\alphain[0,1]} cal(L)(\alpha T)= 10000 circle.stroked.tiny min_{\alphain[0,1]} \alpha^2 + (rho_1+rho_2)\sigma(\alpha^2)$, where $\sigma(x)= x\ln x-x+ 1$.


// If $(A,\mu)$ and $(B,\nu)$ are two cells with unit mass (i.e., $\mu$ and $\nu$ are probability distributions) and the Gromov-Wasserstein distance $d_(GW)((A,\mu),(B,\nu))$ is 50 units (so that $cal(G)(T)$ = 10000, for the optimal strict coupling $T$)
// then for given values of $rho_1,rho_2$, an upper bound for $UGW((A,\mu),(B,\nu))$ is $min_{\alphain[0,1]} cal(L)(\alpha T)= 10000 circle.stroked.tiny min_{\alphain[0,1]} \alpha^2 + (rho_1+rho_2)\sigma(\alpha^2)$, where $\sigma(x)= x\ln x-x+ 1$.

// Computational complications
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// Unbalanced GW is computed using a different algorithm than ordinary GW, and it requires an extra parameter to guide the algorithm, which we now describe.

// Define

// $cal(L)_{\varepsilon}(T) = cal(L)(T) + \varepsilon circle.stroked.tiny KL^{otimes 2}(T \mid \muotimes\nu)$

// and

// $UGW_\varepsilon((X,\mu),(Y,\nu)) = min_{Tin(bb(R)^{<= 0})^{n times m}} cal(L)_\varepsilon(T)$

// We call $cal(L)_{\varepsilon}(T)$ the "entropically regularized cost function". The most important property of $cal(L)_\varepsilon$ is that $\lim_{T_{ij}\to 0^+}\frac{\partial cal(L)(T)}{\partial T_{ij}}\bigg\rvert_{T_{ij=0}}= -infty$, and for $T$ with $T_{ij}=0$, we can always reduce $cal(L)_{\varepsilon}(T)$ by increasing $T_{ij}$ to some very small $\epsilon$. This implies that the global minimum of $cal(L)_\varepsilon(T)$ lies in $(bb(R)^{>0})^{n times m}$, the strict interior of $(bb(R)^{<= 0})^{n times m}$, which lets us apply techniques based on calculus. Therefore, we try to solve this problem instead; we know that in the limit as $\varepsilon\to 0^+$, $UGW_\varepsilon\to UGW$.

// The coefficient $\varepsilon$ has the physical meaning that the 'couplings' will smear each point in the first cell across *every* point in the second cell, at least to some small degree; a bit like an electron cloud, when a point from $X$ is transferred across the coupling to $Y$ it may be highly localized in a certain region of $Y$, but it has nonzero probability mass everywhere in $Y$.

// To choose $\varepsilon$ appropriately, we advise that you experiment with your data set at different values of $\varepsilon$. For sufficiently small $\varepsilon$, the algorithm will tend to diverge due to numerical instability; the most accurate possible result will be given by choosing the smallest possible value of $\varepsilon$ for which all values terminate. One can increase $\varepsilon$ beyond this point, which will tend to make the algorithm converge faster, but to a less accurate answer, so it is a tradeoff to be made based on the size of the dataset to be computed and the precision necessary for the results to be useful. Our experience is that $rho_1,rho_2$ should be at least 20x larger than $\varepsilon$ for the results to be decently accurate, and a higher ratio is probably better.

// Fused Gromov-Wasserstein
// ------------------------

// We let $(X,\mu)$, $(Y,\nu)$ and $cal(G)$ be as before.

// Classical Gromov-Wasserstein treats cells purely geometrically, as shapes. In searching for good alignments between two neurons, it doesn't consider some important information present in cell morphology reconstructions, such as the labels for the soma and dendrite nodes.
// On biological grounds, it is reasonable to argue that a "good alignment" between two neurons should align the soma node to the soma node, align axon to axon, basal dendrites to basal dendrites, and apical dendrites to apical dendrites. Fused Gromov-Wasserstein is a construct
// which modifies classical Gromov-Wasserstein to add a penalty term for transport plans which align nodes of different types. By making the penalty term large, we can bias the search algorithm towards transport plans which reflect the additional information available in the cell structure.

// The formula for the fused GW cost of a transport plan is

// $cal{F}(T) = \alphacal(G)(T) + (1-\alpha)\sum_{ij}C_{ij}T_{ij}$

// and we define

// $FGW_C((X,\mu),(Y,\nu)) = inf_T cal{F}(T)$

// where $C_{ij}$ is a user-supplied penalty matrix, and the value $C_{ij}$ indicates the intrinsic penalty for aligning $X_i$ to $Y_j$.

// In our implementation, the user supplies the penalty for aligning nodes of distinct SWC structure id labels. It is easiest to choose these on relative grounds: for example, if the user wants to impose the constraint that aligning a soma node to a dendrite node is ten times worse than
// aligning a basal dendrite node to an apical node, they can choose the soma-to-dendrite penalty to be 10 and the basal-to-apical penalty to be 1. Once this is done, it remains to choose the coefficient $\alpha$ appropriately.

// Note that if $T^(GW)$ is the optimal transport plan for classical Gromov-Wasserstein and $C$ is a proposed cost matrix, then an upper bound for $FGW_C(X,Y)$ is $cal{F}(T^(GW))$.
// It follows that, if $T^{FGW}$ is the optimal transport plan for fused GW, then

// $cal(G)(T^{FGW}) \leq cal(G)(T^(GW)) + (\frac{1-\alpha}{\alpha})(\sum_{ij}C_{ij}T^(GW)_{ij})$

// One can interpret this inequality as follows: by increasing the term $(\frac{1-\alpha}{\alpha})$, the algorithm will be willing to accept higher distortion in order to better align nodes of similar types. If the user chooses, say, $(\frac{1-\alpha}{\alpha}) = 0.3$,
// then the GW cost of the transport plan $T^(FGW)$ will be at most 30% more than the GW cost of the original transport plan. Thus, our approach to giving an interpretable interface is to allow the user to control how much additional distortion they are willing to accept in the transport plan
// in order to align nodes of the same type.
